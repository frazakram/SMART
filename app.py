import streamlit.web.bootstrap
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
    streamlit.web.bootstrap.run()
