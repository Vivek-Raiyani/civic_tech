import streamlit as st
from agent import agent,validate_image  # Assuming you import tools here

st.set_page_config(page_title="Public Safety Chatbot", page_icon="🛡️", layout="wide")

# CSS styles omitted here for brevity; use your existing styles

st.title("🛡️ Public Safety Chatbot (Indiana)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Two-column layout: left for chat, right for reporting
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Chat")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", placeholder="Type your message here...")
        submit = st.form_submit_button("Send")

    def display_chat():
        for user_msg, bot_msg in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="message user-message">{user_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="message bot-message">{bot_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<br><hr><br>', unsafe_allow_html=True)
    display_chat()

    

    if submit and user_input:
        if user_input.lower() in ["exit", "quit"]:
            st.info("Session ended. Refresh page to start over.")
        else:
            with st.spinner("Thinking..."):
                response = agent.run(user_input)

            st.session_state.chat_history.append((user_input, response))
            st.rerun()

with col2:
    st.header("Report a Safety Issue")

    with st.form(key="report_form"):
        description = st.text_area("Incident Description", help="Briefly describe the safety issue")
        location = st.text_input("Location", help="Specify location of the incident")
        # image_url = st.text_input("Image URL (optional)", help="Link to an image supporting the report")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        submit_report = st.form_submit_button("Report Incident")

    if submit_report:
        if not description or not location:
            st.error("Description and Location are required fields.")
        else:
            if uploaded_file:
                # Call report_safety_issue tool directly
                llmresponse = validate_image(uploaded_file.getbuffer(),description)
                # print(llmresponse)
                response = agent.run(f"{llmresponse}\n\nIncident Description: {description}\nLocation: {location}")
                # result = report_safety_issue(description=description, location=location, image_url=image_url)
                st.success(response)
