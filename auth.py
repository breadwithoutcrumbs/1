import streamlit as st

PASSWORD = "Qwerty96!@"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    pwd = st.text_input("Enter password:", type="password")

    if pwd == PASSWORD:
        st.session_state.authenticated = True
        st.success("Access granted")
        st.rerun()
    elif pwd:
        st.error("Incorrect password")

    st.stop()