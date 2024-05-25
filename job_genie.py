from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


class ANSWERS(BaseModel):
    answer: str = Field(description="The answer to the question")


class JobGenie:
    def __init__(self, openai_api_key: str):
        self.lang_chain_openai = ChatOpenAI(
            openai_api_key=openai_api_key, model="gpt-4")

    def get_job_description_from_file(self) -> str:
        """Reads a job description from a specified text file."""
        with open("job_description.txt", 'r', encoding='utf-8') as file:
            job_description = file.read()
        return job_description

    def get_resume_from_file(self) -> str:
        """Reads a job description from a specified text file."""
        with open("resume.txt", 'r', encoding='utf-8') as file:
            resume = file.read()
        return resume

    def answer_question(self, question: str) -> str:
        job_description = self.get_job_description_from_file()
        resume = self.get_resume_from_file()
        query = f"Answer the following question :{question} based on the job description and resume provided:\n{job_description}\n{resume}"
        template = """
        system:As an AI knowledgeable about career coaching and job application processes, 
        use the provided job description and my resume to answer the question. 
        please provide answers directly addressing the user
        Be specific and provide actionable advice or insights where applicable.: {query}\n{format_instructions}\n
        """
        parser = JsonOutputParser(pydantic_object=ANSWERS)
        prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )
        runnable = prompt | self.lang_chain_openai | parser
        result = runnable.invoke({"query": query})
        return result['answer']


if __name__ == "__main__":
    file_path = "job_description.txt"
    processor = JobGenie(
        openai_api_key=os.getenv("OPENAI_API_KEY"))
    try:
        answer = processor.answer_question(
            "What skills are required for this job?")
        print(answer)
    except FileNotFoundError as e:
        print(e)
