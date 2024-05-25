from flask import Flask, request, jsonify
from flask_cors import CORS
from job_description_processor import JobDescriptionProcessor
from job_genie import JobGenie
from validate_answers import ValidateAnswers
from job_insights import find_matching_skills
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

processor = JobDescriptionProcessor(
    openai_api_key=os.getenv("OPENAI_API_KEY"), mistral_api_key=os.getenv("MISTRAL_API_KEY"))
assistant = JobGenie(
    openai_api_key=os.getenv("OPENAI_API_KEY"))

validate = ValidateAnswers(
    openai_api_key=os.getenv("OPENAI_API_KEY"), mistral_api_key=os.getenv("MISTRAL_API_KEY"))


@app.route('/get-job-matching-insights', methods=['GET'])
def get_job_matching_insights():
    resume_file_path = 'resume.txt'
    try:
        if not os.path.exists(resume_file_path):
            return jsonify({"error": "Resume file not found."}), 404
        with open(resume_file_path, 'r', encoding='utf-8') as file:
            resume_text = file.read()
        job_description = processor.get_job_description_from_file(
            "tech.txt")
        category = processor.job_category(job_description)
        job_skills = processor.extract_skills(job_description, category)
        print("@job_skills", job_skills)
        matching_skills, non_matching_skills, match_percentage = find_matching_skills(
            resume_text, job_skills)

        response = {
            "MatchingSkills": list(matching_skills),
            "SkillsNotInResume": list(non_matching_skills),
            "MatchPercentage": match_percentage
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-questions', methods=['GET'])
def get_questions():
    try:
        job_description = processor.get_job_description_from_file(
            "non_tech.txt")
        questions = processor.generate_questions_from_jd(
            job_description)

        return jsonify(questions)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/submit-answers', methods=['POST'])
def submit_answers():
    # Assuming the format of the json is {'question':'answer'}
    answers = request.get_json()
    result = validate.process_submitted_answers(answers)
    return result


@app.route('/job-genie', methods=['POST'])
def job_genie_answer():
    try:
        data = request.json
        question = data.get(
            'question')
        answer = assistant.answer_question(
            question)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(port=3000)
