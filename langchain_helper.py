from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase 
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt 
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from few_shots import few_shots

load_dotenv()
# import os

def db_chain():
    # LLM MODEL
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7,
        # GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    )
    
    # Setting our db config
    db_user = "root"
    db_password = quote_plus("Azeera@0509")  # encodes @ into %40 automatically
    db_host = "127.0.0.1"
    db_port = "3306"
    db_name = "atliq_tshirts"  # make sure this exists in MySQL
    uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)

    # clean queries from LLM (remove markdown fences)
    def clean_sql(query: str) -> str:
        return (
            query.replace("```sql", "")
                .replace("```", "")
                .replace("SQLQuery:", "")
                .replace("SQLResult:", "")
                .strip()
        )

    # patch db.run to always clean before execution (this is done because the query it generated has some tilde signs which are replaced by that clean sql function)
    orig_run = db.run
    db.run = lambda q, **kwargs: orig_run(clean_sql(q), **kwargs)

    # embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    to_vectorize = ["  ".join(str(v) for v in example.values()) for example in few_shots]

    # vector store
    vector_store = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(
        k=2,
        vectorstore=vector_store
    )

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )
    

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"]  # these variables used in prefix and suffix prompts
    )

    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    print(new_chain)

    return new_chain


if __name__ == "__main__":
    chain = db_chain()
    print(chain)
    print(chain.run("Calculate the total profit we get if we sell all our nike t-shirts"))