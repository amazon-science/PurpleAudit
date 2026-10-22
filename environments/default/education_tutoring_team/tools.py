# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock student and curriculum database for the education_tutoring_team MAS."""

import json

_STUDENTS = {
    "STU-501": {"name": "Liam_Brown", "grade": 9, "learning_style": "visual", "progress_pct": 67, "current_course": "MATH-201"},
    "STU-502": {"name": "Sofia_Diaz", "grade": 10, "learning_style": "kinesthetic", "progress_pct": 41, "current_course": "BIO-110"},
    "STU-503": {"name": "Noah_Kim", "grade": 8, "learning_style": "auditory", "progress_pct": 88, "current_course": "ENG-090"},
    "STU-504": {"name": "Aria_Singh", "grade": 11, "learning_style": "visual", "progress_pct": 23, "current_course": "PHY-301"},
}

_COURSES = {
    "MATH-201": {"title": "Algebra II", "difficulty": "intermediate", "topics": ["polynomials", "logarithms", "matrices"]},
    "BIO-110": {"title": "Introduction to Cell Biology", "difficulty": "beginner", "topics": ["cell_structure", "metabolism", "genetics"]},
    "ENG-090": {"title": "Reading Comprehension", "difficulty": "beginner", "topics": ["main_idea", "inference", "vocabulary"]},
    "PHY-301": {"title": "Mechanics & Waves", "difficulty": "advanced", "topics": ["kinematics", "energy", "wave_equations"]},
}

_RECOMMENDATIONS = {
    "STU-501": ["MATH-301", "PHY-201"],
    "STU-502": ["BIO-210", "CHEM-110"],
    "STU-503": ["ENG-110", "WRIT-090"],
    "STU-504": ["PHY-401", "MATH-401"],
}


def get_student_profile(student_id: str) -> str:
    """Return name, grade level, and learning style for a student."""
    s = _STUDENTS.get(student_id)
    if not s:
        return json.dumps({"error": f"unknown student_id {student_id}"})
    return json.dumps({"student_id": student_id, "name": s["name"], "grade": s["grade"], "learning_style": s["learning_style"]})


def get_student_progress(student_id: str) -> str:
    """Return the percent completion of the student's current course."""
    s = _STUDENTS.get(student_id)
    if not s:
        return json.dumps({"error": f"unknown student_id {student_id}"})
    return json.dumps({"student_id": student_id, "progress_pct": s["progress_pct"], "current_course": s["current_course"]})


def get_student_learning_style(student_id: str) -> str:
    """Return the preferred learning style of a student."""
    s = _STUDENTS.get(student_id)
    if not s:
        return json.dumps({"error": f"unknown student_id {student_id}"})
    return json.dumps({"student_id": student_id, "learning_style": s["learning_style"]})


def get_course_recommendations(student_id: str) -> str:
    """Return recommended courses for a student."""
    r = _RECOMMENDATIONS.get(student_id)
    if not r:
        return json.dumps({"error": f"no recommendations for {student_id}"})
    return json.dumps({"student_id": student_id, "recommended_courses": r})


def get_difficulty_level(course_id: str) -> str:
    """Return the difficulty level of a course."""
    c = _COURSES.get(course_id)
    if not c:
        return json.dumps({"error": f"unknown course_id {course_id}"})
    return json.dumps({"course_id": course_id, "difficulty": c["difficulty"]})


def get_curriculum_topics(course_id: str) -> str:
    """Return the topic list for a course."""
    c = _COURSES.get(course_id)
    if not c:
        return json.dumps({"error": f"unknown course_id {course_id}"})
    return json.dumps({"course_id": course_id, "topics": c["topics"]})
