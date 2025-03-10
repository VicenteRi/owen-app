from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd

# Function for Bond Risk Assessment
def assess_bond_risk(credit_rating, maturity_years, interest_rate):
    risk_score = 0
    
    # Credit rating risk (AAA to D scale)
    rating_risks = {
        "AAA": 1, "AA": 2, "A": 3, "BBB": 4, 
        "BB": 5, "B": 6, "CCC": 7, "CC": 8, 
        "C": 9, "D": 10
    }
    
    risk_score += rating_risks.get(credit_rating.upper(), 5)
    
    # Maturity risk (longer maturity = higher risk)
    if maturity_years > 20:
        risk_score += 3
    elif maturity_years > 10:
        risk_score += 2
    elif maturity_years > 5:
        risk_score += 1
    
    # Interest rate risk (higher rate = potential volatility)
    if interest_rate > 5:
        risk_score += 2
    
    # Determine risk level
    risk_level = "Low"
    if risk_score > 7:
        risk_level = "High"
    elif risk_score > 4:
        risk_level = "Medium"
    
    return risk_level

# Function for Bond Interest Simulator
def calculate_bond_interest(principal, interest_rate, years, compound_frequency):
    # Convert interest rate from percentage to decimal
    interest_rate = interest_rate / 100
    
    # Calculate future value using compound interest formula
    future_value = principal * (1 + interest_rate/compound_frequency)**(compound_frequency * years)
    
    return round(future_value, 2)

# Define the API routes - will be integrated with the main Flask app
def register_finance_routes(app):
    
    @app.route("/bond_risk", methods=['GET'])
    def bond_risk():
        try:
            credit_rating = request.args.get('credit_rating', 'BBB')
            maturity_years = int(request.args.get('maturity_years', 10))
            interest_rate = float(request.args.get('interest_rate', 3.0))
            
            risk_level = assess_bond_risk(credit_rating, maturity_years, interest_rate)
            
            return jsonify({
                "credit_rating": credit_rating,
                "maturity_years": maturity_years,
                "interest_rate": interest_rate,
                "risk_level": risk_level
            })
        except Exception as e:
            print(f"Error in bond_risk: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/bond_interest", methods=['GET'])
    def bond_interest():
        try:
            principal = float(request.args.get('principal', 1000))
            interest_rate = float(request.args.get('interest_rate', 3.0))
            years = int(request.args.get('years', 10))
            compound_frequency = int(request.args.get('compound_frequency', 1))  # 1 for annual, 2 for semi-annual, etc.
            
            future_value = calculate_bond_interest(principal, interest_rate, years, compound_frequency)
            
            return jsonify({
                "principal": principal,
                "interest_rate": interest_rate,
                "years": years, 
                "compound_frequency": compound_frequency,
                "future_value": future_value
            })
        except Exception as e:
            print(f"Error in bond_interest: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/financial_statements", methods=['GET'])
    def financial_statements():
        symbol = request.args.get('symbol', 'AAPL')
        stock = yf.Ticker(symbol)
        
        try:
            # Convert DataFrames to JSON serializable format
            # We need to handle dates and NaN values properly
            def prepare_dataframe(df):
                if df is None or df.empty:
                    return {}
                    
                # Convert DataFrame to dictionary
                df_dict = {}
                for column in df.columns:
                    # Convert column name to string if it's a date
                    if isinstance(column, pd.Timestamp):
                        col_name = column.strftime('%Y-%m-%d')
                    else:
                        col_name = str(column)
                        
                    # Create a dictionary for each column with row indices as keys
                    column_data = {}
                    for idx, value in df[column].items():
                        # Handle NaN and other special values
                        if pd.isna(value):
                            column_data[str(idx)] = None
                        else:
                            try:
                                # Try to convert to native Python types
                                column_data[str(idx)] = float(value) if isinstance(value, (float, int)) else str(value)
                            except:
                                column_data[str(idx)] = str(value)
                                
                    df_dict[col_name] = column_data
                    
                return df_dict
            
            # Get financial statements and convert them to a JSON-compatible format
            try:
                income_statement = prepare_dataframe(stock.income_stmt)
            except:
                income_statement = {}
                
            try:
                balance_sheet = prepare_dataframe(stock.balance_sheet)
            except:
                balance_sheet = {}
                
            try:
                cash_flow = prepare_dataframe(stock.cashflow)
            except:
                cash_flow = {}
            
            return jsonify({
                "symbol": symbol,
                "income_statement": income_statement,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow
            })
        except Exception as e:
            print(f"Error processing financial statements: {str(e)}")
            return jsonify({"error": str(e)}), 500 
