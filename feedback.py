import streamlit as st

def show_feedback():
    st.header("Feedback")
    with st.form("feedback_form"):
        feedback = st.text_area("Your feedback", height=150)
        submit = st.form_submit_button("Submit")
        if submit:
            st.write("Thank you for your feedback!")
