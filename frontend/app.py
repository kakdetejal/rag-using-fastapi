import streamlit as st

import requests

BACKEND_URL = "http://127.0.0.1:8000"

def parse_backend_response(response):
    try:
        data = response.json()
    except requests.JSONDecodeError:
        detail = response.text.strip() or response.reason
        raise RuntimeError(
            f"Backend returned a non-JSON response ({response.status_code}): {detail}"
        )

    if not response.ok:
        detail = data.get("detail") or data
        raise RuntimeError(f"Backend error ({response.status_code}): {detail}")

    return data

st.title(
    "Simple GenAI RAG App"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files
        )

        data = parse_backend_response(response)

        st.success(
            data["message"]
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach backend: {exc}")
    except RuntimeError as exc:
        st.error(str(exc))

question = st.text_input(
    "Ask Question"
)

if st.button("Submit"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",

                json={
                    "question": question
                }
            )

            data = parse_backend_response(response)

            answer = data["answer"]

            st.write("### Answer")

            st.write(answer)
        except requests.RequestException as exc:
            st.error(f"Could not reach backend: {exc}")
        except RuntimeError as exc:
            st.error(str(exc))