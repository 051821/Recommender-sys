from flask import Flask, request, render_template
from model_utiles import load_and_train_model, recommend_restaurants
from jinja2 import TemplateNotFound

app = Flask(__name__)
pipeline, data = load_and_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = {
            'City': request.form['city'],
            'Cuisines': request.form['cuisine'],
            'Country': request.form['country'],
            'Cost': float(request.form['cost']),
            'Rating': float(request.form['rating'])
        }

        recommendations = recommend_restaurants(user_input, data, pipeline)
        print(recommendations)

        table_html = recommendations.to_html(classes='data', index=False, border=0)

        return render_template('results.html', table_html=table_html)

    except TemplateNotFound as e:
        return f"Template not found error: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
