import streamlit as st

st.title("Simple Test App")
st.write("If you can see this, Streamlit is working!")

# Test basic functionality
st.write("Current working directory:")
import os
st.write(os.getcwd())

st.write("Python path:")
import sys
for path in sys.path[:5]:  # Show first 5 paths
    st.write(f"- {path}")