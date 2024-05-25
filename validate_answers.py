from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()


class VALIDATE(BaseModel):
    result: str = Field(description="Return either correct or incorrect")


class ValidateAnswers:
    def __init__(self, openai_api_key: str, mistral_api_key: str):
        self.lang_chain_openai = ChatOpenAI(
            openai_api_key=openai_api_key, model="gpt-4")
        self.lang_chain_mistral = ChatMistralAI(
            model="mistral-large-latest", temperature=0, mistral_api_key=mistral_api_key)

    def process_submitted_answers(self, answers_json: dict) -> dict:
        parser = None
        count = 0
        for question, answer in answers_json.items():
            validate_query = f"Validate  the answer: {answer} with respect to the question : {question}. "
            parser = JsonOutputParser(pydantic_object=VALIDATE)
            prompt = PromptTemplate(
                template="system:You are an assistant knowledgeable in validating given questions from given question.\n{format_instructions}\n{query}\n",
                input_variables=["query"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()},
            )
            runnable = prompt | self.lang_chain_openai | parser
            result = runnable.invoke({"query": validate_query})
            if (result['result'] == 'correct'):
                count += 1
        return "Your response has been submitted successfully"


if __name__ == "__main__":
    processor = ValidateAnswers(
        openai_api_key=os.getenv("OPENAI_API_KEY"), mistral_api_key=os.getenv("MISTRAL_API_KEY"))

    answers_json = {
        "What does JDBC stand for?": "Sreeram",
        "Which method is used to start a thread in Java?": "start()",
        "What is the default transaction isolation level in JDBC?": "TRANSACTION_READ_COMMITTED",
        # Incorrect for demonstration
        "How can you retrieve the auto-generated keys after an INSERT statement in SQL?": "Using getGeneratedKeys() method of Statement object.",
        "What is the main difference between 'INNER JOIN' and 'LEFT JOIN' in SQL?": "INNER JOIN returns rows when there is at least one match in both tables. LEFT JOIN returns all rows from the left table, and the matched rows from the right table; if there is no match, the result is NULL on the right side."  # Incorrect for demonstration
    }
    validated = processor.process_submitted_answers(answers_json)
    print(validated)
