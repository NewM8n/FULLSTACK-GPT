import json
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

import streamlit as st
from datetime import datetime

if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0

st.session_state["run_count"] += 1

start_time = datetime.now()
print(
    f"\n\033[43mSTART Exec[{st.session_state['run_count']}]: {start_time.strftime('%H:%M:%S.%f')} =========================================\033[0m"
)

st.set_page_config(
    page_title="QuizGPT",
    page_icon="☘️",
)

st.title("QuizGPT Turbo")


with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        print(st.session_state["api_key"])

    if st.button(":red[Reset API_KEY]"):
        reset_api_key()

    api_key = api_key_input.text_input(
        ":blue[OpenAI API_KEY]",
        value=st.session_state["api_key"],
        key="api_key_input",
    )

    if api_key != st.session_state["api_key"]:
        st.session_state["api_key"] = api_key
        st.rerun()

    # print(api_key)

    st.divider()
    st.markdown(
        """
        GitHub 링크: "https://github.com/NewM8n/FULLSTACK-GPT"
        """
    )

if not api_key:
    st.warning("Please provide an :blue[OpenAI API Key] on the sidebar.")

else:
    try:
        if "quiz_subject" not in st.session_state:
            st.session_state["quiz_subject"] = ""

        if "quiz_submitted" not in st.session_state:
            st.session_state["quiz_submitted"] = False

        def set_quiz_submitted(value: bool):
            st.session_state.update({"quiz_submitted": value})

        @st.cache_data(show_spinner="퀴즈를 만들고 있어요...")
        def run_quiz_chain(*, subject, count, difficulty):
            chain = prompt | llm

            return chain.invoke(
                {
                    "subject": subject,
                    "count": count,
                    "difficulty": difficulty,
                }
            )

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                """
            #### 퀴즈 ㄱㄱ?
            """
            )
        with col2:

            def reset_quiz():
                st.session_state["quiz_subject"] = ""
                run_quiz_chain.clear()

            # 제대로 동작하지 않음. => 수정 필요
            if st.button(":red[퀴즈 초기화]"):
                reset_quiz()
                set_quiz_submitted(False)

        with st.form("quiz_create_form"):

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                quiz_subject = st.text_input(
                    ":blue[주제]",
                    placeholder="무슨 퀴즈를 만들까?",
                    value=st.session_state["quiz_subject"],
                    # label_visibility="collapsed",
                )

            with col2:
                quiz_count = st.number_input(
                    ":blue[개수]",
                    placeholder="개수",
                    value=10,
                    min_value=2,
                    # label_visibility="collapsed",
                )

            with col3:
                quiz_difficulty = st.selectbox(
                    ":blue[레벨]",
                    ["1", "2", "3", "4", "5"],
                    # label_visibility="collapsed",
                )

            st.form_submit_button(
                "**:blue[퀴즈 만들기 시작]**",
                use_container_width=True,
                on_click=set_quiz_submitted,
                args=(False,),
            )

        function = {
            "name": "create_quiz",
            "description": "function that takes a list of questions and answers and returns a quiz",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                },
                                "answers": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "answer": {
                                                "type": "string",
                                            },
                                            "correct": {
                                                "type": "boolean",
                                            },
                                        },
                                        "required": ["answer", "correct"],
                                    },
                                },
                            },
                            "required": ["question", "answers"],
                        },
                    }
                },
                "required": ["questions"],
            },
        }
        # ChatOpenAI model 정보
        # - https://platform.openai.com/docs/models/gpt-3-5-turbo
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-3.5-turbo-0125",
            temperature=0.1,
        ).bind(
            function_call="auto",
            functions=[
                function,
            ],
        )

        prompt = PromptTemplate.from_template(
            """            
            Please create a quiz based on the following criteria:

            Topic: {subject}
            Number of Questions: {count}
            Difficulty Level: Level-{difficulty}/5
            Language: Korean

            The quiz should be well-structured with clear questions and correct answers.
            Ensure that the questions are relevant to the specified topic and adhere to the selected difficulty level.
            The quiz format should be multiple-choice,
            and each question should be accompanied by four possible answers, with only one correct option.
            """,
        )

        if quiz_subject:
            response_box = st.empty()
            response = run_quiz_chain(
                subject=quiz_subject,
                count=quiz_count,
                difficulty=quiz_difficulty,
            )
            response = response.additional_kwargs["function_call"]["arguments"]
            response = json.loads(response)

            generated_quiz_count = len(response["questions"])

            with st.form("quiz_questions_form"):
                solved_count = 0
                correct_count = 0
                answer_feedback_box = []
                answer_feedback_content = []

                for index, question in enumerate(response["questions"]):
                    st.write(f"{index+1}. {question['question']}")
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                        label_visibility="collapsed",
                        key=f"[{quiz_subject}_{quiz_count}_{quiz_difficulty}]question_{index}",
                    )

                    answer_feedback = st.empty()
                    answer_feedback_box.append(answer_feedback)

                    if value:
                        solved_count += 1

                        if {"answer": value, "correct": True} in question["answers"]:
                            answer_feedback_content.append(
                                {
                                    "index": index,
                                    "correct": True,
                                    "feedback": "정답! :100:",
                                }
                            )
                            correct_count += 1
                        else:

                            answer_feedback_content.append(
                                {
                                    "index": index,
                                    "correct": False,
                                    "feedback": "다시 도전 :sparkles:",
                                }
                            )


                is_quiz_all_submitted = solved_count == generated_quiz_count

                if is_quiz_all_submitted:
                    for answer_feedback in answer_feedback_content:
                        index = answer_feedback["index"]
                        with answer_feedback_box[index]:
                            if answer_feedback["correct"]:
                                st.success(answer_feedback["feedback"])
                            else:
                                st.error(answer_feedback["feedback"])

                st.divider()

                result = st.empty()

                st.form_submit_button(
                    (
                        "**:blue[제출하기]**"
                        if solved_count < generated_quiz_count
                        else (
                            "**:blue[:100: 축하합니다]**"
                            if correct_count == generated_quiz_count
                            else "**:blue[다시 도전]**"
                        )
                    ),
                    use_container_width=True,
                    disabled=correct_count == generated_quiz_count,
                    # on_click=lambda: setattr(st.session_state, "submitted", True), 동일함.
                    on_click=set_quiz_submitted,
                    args=(True,),
                )

                if st.session_state["quiz_submitted"]:

                    if not is_quiz_all_submitted:
                        result.error(
                            f"퀴즈를 모두 풀고 제출해 주세요. ( 남은 퀴즈 개수: :red[{generated_quiz_count - solved_count}] / 답변한 퀴즈 개수: :blue[{solved_count}] )"
                        )
                    else:
                        result.subheader(
                            f"결과: :blue[{correct_count}] / {generated_quiz_count}"
                        )

                    if correct_count == generated_quiz_count:
                        for _ in range(3):
                            st.balloons()

    except Exception as e:

        e_str = str(e).lower()
        match = re.search(r"(api)(_|-|\s)(key)", e_str)
        if match:
            st.error("API_KEY 를 확인해 주세요.")

        st.expander("Error Details", expanded=True).write(f"Error: {e}")

        if "response" in locals():
            response_box.json(response)


end_time = datetime.now()
elapsed_time = end_time - start_time
elapsed_seconds = elapsed_time.total_seconds()
print(
    f"\n\033[43mEND Exec[{st.session_state['run_count']}]: {elapsed_seconds}s / {end_time.strftime('%H:%M:%S.%f')} =========================================\033[0m"
)