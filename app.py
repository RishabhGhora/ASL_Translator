###################################################
# Outlines the UI created with Dash               #
###################################################

# Load the necessary modules
import datetime
import dash
import dash_table
from dash.dependencies import Input, Output, State
from dash_extensions import Download
import dash_core_components as dcc
import dash_html_components as html
from translate_video import get_frames, predict_frames, display_data

# Global variables 
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, title='ASL Translator', external_stylesheets=external_stylesheets, suppress_callback_exceptions=True, prevent_initial_callbacks=True)

app.layout = html.Div([

    ### code for the HEADER
    html.Div([
        html.H1(
            "ASL Translator",
            id='title-header',
            style={
                'textAlign': 'center'
            }
        ), html.H5(
            "A CS 4476 Project",
            style={
                'textAlign': 'center'
            }
        ), html.P(
            "This project attempts to translate a minute long ASL alphabet video to english letters",
            style={
                'textAlign': 'center'
            }
        )
    ],  style={
            'textAlign': 'center',
            'padding-top': 15,
            'padding-bottom': 15
        }),
    ### end code for the HEADER
  
    html.Div([
        html.Button('Translate video', id='btn', n_clicks=0),
        dcc.Loading(
            id="loading-1",
            type="circle",
            children=html.Div(id="video-container")
    )], style={
            'textAlign': 'center',       
    })
])

@app.callback(Output('video-container', 'children'),
              Input('btn', 'n_clicks'))
def start_translation(n_clicks):
    """
    Start after button is pressed 
    Gets frames from video, crops frames, 
    predicts on frames, displays results 
    n_clicks: number of times button is pressed 
    """
    if n_clicks != 0:
        get_frames('signs.mp4')
        predictions, filenames = predict_frames()
        images, translations = display_data(predictions, filenames)
        output = []
        output.append(html.Div([
            html.Iframe(src="https://www.youtube.com/embed/6_gXiBe9y9A", width=560, height=315)
        ], style={
            'textAlign': 'center', 'padding-top': '5%'
        }))
        for i in range(len(images)):
            output.append(format_image(images[i], translations[i]))
        table = []
        for i in range(len(predictions)):
            table.append({'Filenames': filenames[i], 'Predictions': map_characters[predictions[i]]})
        output.append(html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{'name': i, 'id': i} for i in ['Filenames', 'Predictions']],
            data=table,
            editable=False,
        )], style={
            'height': 700, 'width': '60%', 'margin': 'auto', 'align-content': 'center', 'padding': '10px',
        }))
        return output

def format_image(image, translation):
    """
    Creates Div from Image and translation
    image: jpeg image opened with Pillow
    translation: letter predicted with model
    """
    return html.Div([
        html.H5(translation, id='translation'),
        html.Img(src=image, style={'padding-top': '1%', 'width': 300, 'height': 200}),
        html.Br(),html.Br(),
    ], style={
        'text-align': 'center'
    })

if __name__ == '__main__':
    app.run_server(debug=True)