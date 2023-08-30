from django.shortcuts import render
from .utils import load_data, get_future_closes
from django.views import View
import plotly.express as px
from plotly.offline import plot
import tensorflow as tf
from itertools import cycle
import pickle
# Create your views here.
lstm = tf.keras.models.load_model("bitcoin_app\\lstm_model")
xgb_reg =  pickle.load(open("bitcoin_app\\xgb_model.pkl", "rb"))
class Index(View):
    def get(self, request):
        limit = 200
        future_days = 0
        data = load_data('BTC', 'USD', limit)
        fig = px.line(data, x = 'date', y = 'close')
        line_plot = plot(fig, output_type="div")
        context = {'plot_div': line_plot,
                   'limit': limit,
                   'future_days': future_days,
                   'algorithm': 'lstm'}
        return render(request, 'bitcoin_app/index.html', context)
    def post(self, request):
        limit = int(request.POST["limit"])
        data = load_data('BTC', 'USD', limit)
        future_days = int(request.POST["future-days"])
        if future_days != 0:
            algorithm = request.POST["algorithm"]
            if algorithm == "lstm":
                model = lstm
                time_step = 7
            else:
                model = xgb_reg
                time_step = 26
            future_closes = get_future_closes(model, data, future_days, time_step)
            names = ["Close Price", "Predicted Close Price"]
            names = cycle(names)
            fig = px.line(future_closes,x=future_closes['date'], y=[future_closes['close'],future_closes['predicted_close']],
                labels={'value':'close','date': 'Date'})

            fig.for_each_trace(lambda t:  t.update(name = next(names)))
            line_plot = plot(fig, output_type="div")
            context = {'plot_div': line_plot,
                       'limit': limit,
                       'future_days': future_days,
                       'algorithm': algorithm}
            return render(request, 'bitcoin_app/index.html', context)
        else:
            fig = px.line(data, x = 'date', y = 'close')
            line_plot = plot(fig, output_type="div")
            context = {'plot_div': line_plot,
                       'limit': limit,
                       'future_days': future_days,
                       'algorithm': 'lstm'}
            return render(request, 'bitcoin_app/index.html', context)
        
