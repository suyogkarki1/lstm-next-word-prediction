"""Streamlit version of the NextWord keyboard (for Streamlit Community Cloud).

The iPhone keyboard in frontend/ runs as a custom Streamlit component. Each request it
sends ({id, kind, text}) triggers a rerun; Python answers it and passes the result back to
the component as an argument, which resolves the matching request in the browser.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from nextword.model import Predictor

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / 'artifacts' / 'lstm_next_word.pt'

st.set_page_config(page_title='NextWord Keyboard', page_icon='⌨️', layout='wide')
st.markdown("""
<style>
  #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
  .block-container { padding: 0.5rem 1rem 0 !important; max-width: 100% !important; }
  iframe { display: block; }
</style>
""", unsafe_allow_html=True)

keyboard = components.declare_component('nextword_keyboard', path=str(ROOT / 'frontend'))


@st.cache_resource
def load_predictor():
    return Predictor(CHECKPOINT)


def answer(request):
    predictor = load_predictor()
    text = request.get('text', '')
    if request.get('kind') == 'continue':
        words, finished = predictor.generate(text)
        data = {'words': words, 'finished': finished}
    else:
        data = predictor.suggest(text, k=3)
    return {'id': request['id'], 'data': data}


# the component's latest value (its request) is in session state before it is drawn again
request = st.session_state.get('keyboard')
response = answer(request) if isinstance(request, dict) and 'id' in request else None
keyboard(response=response, key='keyboard', default=None)
