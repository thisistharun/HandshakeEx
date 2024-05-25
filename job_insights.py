import spacy

# Load the spaCy medium model
nlp = spacy.load("en_core_web_md")


def calculate_similarity(skill1, skill2):
    return nlp(skill1).similarity(nlp(skill2))


def find_matching_skills(resume_text, job_skills, similarity_threshold=0.7):
    resume_doc = nlp(resume_text.lower())
    matching_skills = set()
    non_matching_skills = set(job_skills)

    for job_skill in job_skills:
        job_skill_doc = nlp(job_skill.lower())
        for token in resume_doc:
            if token.is_alpha and calculate_similarity(job_skill_doc.text, token.text) >= similarity_threshold:
                matching_skills.add(job_skill)
                non_matching_skills.discard(job_skill)
                break

    match_percentage = (len(matching_skills) /
                        len(job_skills)) * 100 if job_skills else 0
    return matching_skills, non_matching_skills, match_percentage
