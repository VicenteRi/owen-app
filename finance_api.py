from flask import Flask, jsonify, request
import yfinance as yf

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
    
    @app.route("/bond_interest", methods=['GET'])
    def bond_interest():
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
    
    @app.route("/financial_statements", methods=['GET'])
    def financial_statements():
        symbol = request.args.get('symbol', 'AAPL')
        stock = yf.Ticker(symbol)
        
        try:
            income_statement = stock.financials.to_dict()
            balance_sheet = stock.balance_sheet.to_dict()
            cash_flow = stock.cashflow.to_dict()
            
            return jsonify({
                "symbol": symbol,
                "income_statement": income_statement,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500 