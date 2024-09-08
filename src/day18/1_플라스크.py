

from flask import Flask
import pandas as pd
import json
app = Flask( __name__ )

from flask_cors import CORS
CORS( app ) # 모든 경로에 대해 CORS 허용

@app.route( "/" )
def index() :
    CB_file = pd.read_csv('CoffeeBean_2.csv', encoding='cp949', engine='python')
    jsonResult = CB_file.to_json(orient='records', force_ascii=False)

    result = json.loads(jsonResult)
    return result


if __name__ == "__main__" :
    app.run( )