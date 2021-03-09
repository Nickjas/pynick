import streamlit as st
from pages import home,data_info, reg_preprocessing,prediction, reg_training, reg_model_analysis,cls_preprocessing,cls_training,cls_model_analysis,cluster_preprocessing,cluster_training,cluster_model_analysis

from pathlib import Path
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher
import base64
from PIL import Image
#session============================================================================
class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state
#====loading image=======================================================================================



def load_nav_image(image_path):
    image_eido = Image.open(image_path)
    st.sidebar.image(image_eido, use_column_width=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def load_header_image(image_path):
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(image_path))
    st.markdown(
        header_html, unsafe_allow_html=True,
    )
    st.markdown("---")   
    #========================================================================
PAGES = {
    "Home": home,
    "DataInfo": data_info,
    "Preprocessing": (reg_preprocessing,cls_preprocessing,cluster_preprocessing),
    "Training" : (reg_training, cls_training,cluster_training),
    "Model Analysis": (reg_model_analysis, cls_model_analysis,cluster_model_analysis),
    "Prediction and Save": prediction,
    #"Backward Analysis": backward_analysis,
}

IMAGE_FOLDER = Path("images/")

def run():
    state = _get_state()
    #st.set_page_config(
      #  page_title="EidoData App",
       # page_icon=':shark:',
       # layout="centered",
       # initial_sidebar_state='expanded'
   # )
    load_nav_image(IMAGE_FOLDER/'NI.jpg')
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection == "Home":
        #load_nav_image(IMAGE_FOLDER/'nicklogo.png')
        load_header_image(IMAGE_FOLDER/'im1.jpg')
        
        try:
            state_df,task = PAGES[selection].write(state)
            state.df, state.task = state_df,task
        except:
            st.header("Please Upload Csv or Excel File first!")
            st.stop()
    if selection == "DataInfo":
        data_info.write(state.df)
    
    if selection == "Preprocessing":
        if state.task == "Regression":
            reg_preprocessing.write(state)
        elif state.task =="Classification":
            cls_preprocessing.write(state)
        else:
            cluster_preprocessing.write(state)

    if selection == "Training":
        if state.task == "Regression":
            reg_training.write(state)
        elif state.task =="Classification":
            cls_training.write(state)
        else:
            cluster_training.write(state)
    if selection == "Model Analysis":
        if state.task == "Regression":
            reg_model_analysis.write(state)
        elif state.task =="Classification":
            cls_model_analysis.write(state)
        else:
            cluster_model_analysis.write(state)
    if selection == "Prediction and Save":
        prediction.write(state)
        
    if selection == "Backward Analysis":
        if state.task == "Regression":
            prediction.write(state)
        else:
            st.header("Only Support for Regression Task!")
    st.write(state.__dict__)
    state.sync()


if __name__ == '__main__':
    run()
