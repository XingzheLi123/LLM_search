import requests
import pandas as pd
import os
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTS
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.callbacks import get_openai_callback

class Reader:
    def __init__(self, index_name, email, target = None): #target companies
        self.headers = {'User-Agent': email}
        load_dotenv()
        self.openai_api = os.environ.get("openai-api")
        self.pine_api = os.environ.get('pinecone-api')
        self.pine_env = os.environ.get("pinecone-env")
        pinecone.init(
            api_key=self.pine_api,  
            environment=self.pine_env  
        )
        self.index_name = index_name
        self.target = target #Optional, target list of companies

    def get_all_ciks(self, year): # year being the last full financial year you wish to include. The function grabs a list of company data from SEC API.
        companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json", headers = self.headers)
        companies = pd.DataFrame.from_dict(companyTickers.json(), orient = 'index')
        targets = list(set(self.target['ticker'])) 
        if len(targets) != 0:
            companies = companies[companies['ticker'].isin(targets)] 
        current = pd.read_csv("years.csv")
        not_needed = list(set(current[current['latest'] >= year]['ticker'])) #Already up-to-date companies
        companies = companies[~companies['ticker'].isin(not_needed)] 
        companies['cik_str'] = companies['cik_str'].astype(str).str.zfill(10)
        #companies = companies.head(200) #Optional, downloading in chunks
        return companies
    
    def get_all_10K(self, cik, ticker): #Download all the relevant 10K forms (annual reports) of a company. 
        filingMetadata = requests.get(
            f'https://data.sec.gov/submissions/CIK{cik}.json',
            headers=self.headers
            )
        allData = pd.DataFrame.from_dict(filingMetadata.json()['filings']['recent'])
        form_10ks = allData[allData['primaryDocDescription'] == '10-K']
        form_10ks = form_10ks.sort_values(by = 'filingDate', ascending = False)
        form_10ks = form_10ks.head(1) #How many you want
        for index, row in form_10ks.iterrows():

            doc = row['primaryDocument']
            accessionNumber = row['accessionNumber'].replace('-', '')
            file = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessionNumber}/{doc}", headers = self.headers)
            date = row['reportDate'][:4]
            if not os.path.exists(f'./Annual Reports/{ticker}'):
                os.makedirs(f'./Annual Reports/{ticker}')
            open(f'./Annual Reports/{ticker}/{date}.html', 'wb').write(file.content)

    def download_all(self, year = 2022): #Download all the relevant 10K forms of all companies.
        df = self.get_all_ciks(year)
        i = 0
        l = len(df['title'])
        for index, row in df.iterrows():
            self.get_all_10K(row['cik_str'], row['ticker'])
            i+=1
            print(f'{i}/{l}')

    def upload(self, ticker): #Tokenize and upload a companies reports to Pinecone
        #embeddings = OpenAIEmbeddings(openai_api_key = self.openai_api) 
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        years = []
        latest = 0
        for file in os.listdir(f'./Annual Reports/{ticker}'):
            try:
                d = UnstructuredHTMLLoader(f"./Annual Reports/{ticker}/{file}").load()
            except:
                continue
            year = file[:-5]
            if int(year) > latest:
                latest = int(year)
            text_splitter = RCTS(chunk_size = 900, chunk_overlap = 40)
            text = text_splitter.split_documents(d)
            docsearch = Pinecone.from_texts([t.page_content for t in text], embeddings, index_name=self.index_name, metadatas = [{'source': f'{ticker}_{year}'} for t in text])
            years.append(year)
        return {'ticker': ticker, 'year': years, 'latest': latest}
    
    def upload_all(self): #Upload all downloaded reports to Pinecone
        tickers = []
        i = 0
        l = len(os.listdir('./Annual Reports'))
        for ticker in os.listdir('./Annual Reports'):
            try:
                dict = self.upload(ticker)
                tickers.append(dict)
            except:
                continue
            i+=1
            print(f'{i}/{l}')
        df = pd.DataFrame.from_dict(tickers)
        df.to_csv('years.csv', mode = 'a', index = False, header = False)
    
    def search(self, company, ticker, year, value): #Searching for any value in any year of reports

        embeddings = OpenAIEmbeddings(openai_api_key = self.openai_api)
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_kwargs = {'device': 'cpu'}
        # encode_kwargs = {'normalize_embeddings': False}
        # embeddings = HuggingFaceEmbeddings(
        #     model_name=model_name,
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )
        llm_ins = ChatOpenAI(temperature = 0, openai_api_key = self.openai_api, model_name = "gpt-3.5-turbo-16k")
        #llm_ins = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length": 1024})
        index = pinecone.Index(self.index_name)

        vectorstore = Pinecone(
            index, embeddings.embed_query, 'text'
        )
        metadata = {'company': company}
        #metadata = {'source': f'{ticker}_{year}'}
        prompt = f"""
        The provided context is the annual report of {company} in of the financial year ended in {year}. What was the value of {value} this year for {company}? Note that this value maybe negative. Values with parentheses around them means it is negative. 
        For example, $(11,022) million is -$11,022 million.

        Answer with just a value, like "-$5,223 million" or "1,203 million shares"
        Output:
        """
        retrieval = RetrievalQA.from_chain_type(llm = llm_ins, chain_type = "stuff", retriever = vectorstore.as_retriever(search_kwargs = {'k': 20, 'filter': metadata})) 
        with get_openai_callback() as cb:
            answer = retrieval.run(prompt)
            print("initial: ", cb)
        print(answer)
        format = """
        Convert an input to just a number in millions with no additional text. For example, if the input is "-$5,223 million", output -5223; if the input is "1.203 billion shares", output 1203. 
        If the input is not a number, output 'NaN'.
        Input: {input}
        Output: 
        """
        format_prompt = PromptTemplate(input_variables = ['input'], template = format)
        formatting = LLMChain(llm = llm_ins, prompt=format_prompt)
        with get_openai_callback() as cb:
            formatted_answer = formatting.run(answer)
            print('formatting: ', cb)
        return formatted_answer
    



            

    
    
