import gradio as gr
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot




with gr.Blocks() as demo:
    df_imported_var = gr.State(pd.DataFrame)
    df_prophet = gr.State(pd.DataFrame)
    gr.Markdown("# Forecast your time series data with prophet")
    with gr.Tab("Load Data"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Load a csv file of time series data. The file must contain a date and value column.")
                file_input = gr.File()
                file_input_btn = gr.Button("Load data")
            with gr.Column():
                initial_dataframe = gr.DataFrame(type="pandas", height=250)
        with gr.Row():
            with gr.Column() as radios:
                gr.Markdown('### Select the columns to use in your forecast')
                date_radio = gr.Radio(visible=False)
                value_radio = gr.Radio(visible=False)
                radios_submit = gr.Button(visible=False)
            with gr.Column():
                prophet_dataframe = gr.DataFrame(height=250)
        with gr.Row():
            gr.Markdown('### Now switch to the "Forecast" tab to generate a forecast using your imported data!')

    def load_data(file, df_input):
        df = pd.read_csv(file.name)
        cols = df.columns.to_list()
        return {
            initial_dataframe:df,
            date_radio:gr.Radio(choices=cols, type='value', label='Select date column:', visible=True),
            value_radio:gr.Radio(choices=cols, type='value', label='Select value column:', visible=True),
            radios_submit:gr.Button('Submit Selections', visible=True),
            df_imported_var: df}

    def create_prophet_data(df_input, date_col, value_col):
        df = df_input.copy()
        df = df[[date_col, value_col]]
        df = df.rename(columns = {date_col:'ds', value_col:'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.groupby('ds').sum().reset_index()
        return {prophet_dataframe:df, df_prophet:df}

    file_input_btn.click(load_data, [file_input, df_imported_var], [initial_dataframe, date_radio, value_radio, radios_submit, df_imported_var])
    radios_submit.click(create_prophet_data, [df_imported_var, date_radio, value_radio], [prophet_dataframe, df_prophet])

    with gr.Tab("Forecast"):
        gr.Markdown("### Enter any changes to your forecast parameters, then select Submit")
        with gr.Row():
            with gr.Accordion(label='Forecast options:', open=True) as acc_options:
                with gr.Column():
                    future_periods = gr.Number(label='Periods to forecast:', value=24, precision=0, minimum=1, maximum=10000, step=1)
                    changepoint_prior_scale = gr.Number(label='Changepoint prior scale:', value=0.05, precision=2, minimum=.01, step = .01)
                    seasonality_prior_scale = gr.Number(label='Seasonality prior scale:', value=10, precision=2, minimum=.01, maximum=10, step=.01)
                    seasonality_mode = gr.Radio(label='Seasonality mode:', value='additive', choices=['additive', 'multiplicative'])
                    forecast_submit = gr.Button('Submit')
        with gr.Row():
            forecast_plot = gr.Plot()
        with gr.Row():
            component_plot = gr.Plot()

    def run_prophet(df_in, future_pers, changepoint_prior, seasonality_prior, seasonality_mode):
        m = Prophet(changepoint_prior_scale=changepoint_prior, seasonality_prior_scale=seasonality_prior, seasonality_mode=seasonality_mode)
        m.fit(df_in)
        future = m.make_future_dataframe(periods=future_pers)
        forecast = m.predict(future)
        forecast_plot = m.plot(forecast)
        a = add_changepoints_to_plot(forecast_plot.gca(), m, forecast)
        components_plot = m.plot_components(forecast) 

        return [forecast_plot, components_plot, gr.Accordion(label='Forecast options:',open=False)]

    forecast_submit.click(run_prophet,
                          [df_prophet, future_periods, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode],
                          [forecast_plot, component_plot, acc_options])


if __name__ == "__main__":
    demo.launch()