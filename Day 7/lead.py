import streamlit as st
import requests

# Set your webhook URL here
WEBHOOK_URL = "https://dhanvarshini.app.n8n.cloud/webhook-test/lead"

# Streamlit UI
st.title("Lead Capture Form")

with st.form("lead_form"):
    name = st.text_input("Name")
    email = st.text_input("Mail ID")
    contact = st.text_input("Contact Number")
    services = st.multiselect(
        "Services Looking For",
        ["Product Demo", "Website Creation", "Augmentation Services"]
    )
    
    submit = st.form_submit_button("Submit")

    if submit:
        if not name or not email or not contact or not services:
            st.warning("Please fill all the fields.")
        else:
            # Prepare data payload
            data = {
                "name": name,
                "email": email,
                "contact": contact,
                "services": services
            }

            # Send to webhook
            try:
                response = requests.post(WEBHOOK_URL, json=data)
                if response.status_code == 200:
                    st.success("Data submitted successfully!")
                else:
                    st.error(f"Failed to submit data. Status Code: {response.status_code}")
            except Exception as e:
                st.error(f"Error occurred: {e}")
