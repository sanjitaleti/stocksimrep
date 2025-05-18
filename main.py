# === FUNCTION TO FETCH STOCK DATA AND GENERATE A CHART ===
def generate_stock_chart(symbol: str):
    try:
        data = yf.download(symbol, period="1y").reset_index()
        if data.empty:
            st.error(f"❌ No data found for {symbol}.")
            return None

        # Convert 'Date' column to datetime objects
        data['Date'] = pd.to_datetime(data['Date'])

        # Define the color gradient
        gradient_colors = ['#FFD700', '#FFA500', '#FF4500'] # Example: Gold to Orange to Red

        # Create Altair chart with gradient line
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X('Date:T', title='Date'), # T specifies time type
            y=alt.Y('Close:Q', title='Close Price'),
            color=alt.Color('Close:Q', scale=alt.Scale(range=gradient_colors)),
            tooltip=['Date:T', 'Close:Q']
        ).properties(
            title=f'{symbol.upper()} Stock Price (1 Year)'
        ).interactive()

        return chart

    except Exception as e:
        st.error(f"❌ Chart Generation Error: {str(e)}")
        return None
