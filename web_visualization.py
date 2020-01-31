from flask import Flask, render_template, request
from visualize import plot_visualizer
app = Flask(__name__)


"""  Oppgave 6.4:
@app.route("/")
@app.route("/home")
def home():
    oppgave 6.4:
    results = plot_visualizer("svc", "age", "pressure")
    return render_template("plot6_4.html", accuracy= results)
"""

@app.route("/")
@app.route("/home")
def home():
    """renders home page 
    """
    return render_template("home.html")

@app.route("/plot", methods=["POST"])
def plot():
    """Function to show accuracy and plot, selected at home page. 
    """
    classifier = request.form.get("Classifier")
    feature_1 = request.form.get("Feature 1")
    feature_2 = request.form.get("Feature 2")
    results = plot_visualizer(classifier, feature_1, feature_2)
    img_filename = "static/" + classifier + feature_1 + feature_2 + ".png"
    return render_template("plot.html", accuracy = results, filename = img_filename)


if __name__ == "__main__":
     app.run(debug=False)
