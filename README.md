# LLM_search
Automated pipeline for searching financial data in annual reports using LLM

# How to Use
Clone to local and from search import Reader.
Run Reader.download_all() to download annual reports. You might want to rewrite bits of Reader to select the annual reports appropriate for your use case. Currently, it uses SEC API to download 10K forms (annual reports) from any company with SEC filings.
There is already a few downloaded for evaluations.
Run Reader.upload_all() to embed the downloaded reports and store it into a Pinecone data base. Again, reconfigure to suit your own use case.
Run Reader.search(company, year, value) to find the financial data "value" of "company" in the annual report of "year". Adjust the prompt to your need.

# Warning!
The performance is wildly inconsistent depending on the value you ask for, the report you are looking through, the model you use, and just general luck. I recommend first using the evaluation notebook to evaluate based on your specific choices.

