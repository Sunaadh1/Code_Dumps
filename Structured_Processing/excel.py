import pandas as pd
import os
import numpy as np
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent



os.environ["OPENAI_API_KEY"] = "Your API key"

excel_file_path = 'SampleData.xlsx'
df = pd.read_excel(excel_file_path)
# print(df.head(3))
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

prompt=input("Please enter the prompt : ")
agent.run(prompt)
