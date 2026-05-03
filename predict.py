import pickle
import pandas as pd

# LOAD
model = pickle.load(open("model.pkl", "rb"))
route_encoder, day_encoder, phase_encoder = pickle.load(open("encoders.pkl", "rb"))

def predict_all(route, day, phase):
    try:
        route_enc = route_encoder.transform([route])[0]
        day_enc = day_encoder.transform([day])[0]
        phase_enc = phase_encoder.transform([phase])[0]

        X = pd.DataFrame([[route_enc, day_enc, phase_enc]],
                         columns=["Route_enc", "Day_enc", "Phase_enc"])

        pred = model.predict(X)[0]

        revenue = round(pred[0], 2)
        expense = round(pred[1], 2)
        profit = round(pred[2], 2)

        return {
            "Revenue": revenue,
            "Expense": expense,
            "Profit": profit
        }

    except Exception as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    route_id = input("Enter Route ID: ")
    day = input("Enter Day: ")
    phase = input("Enter Phase: ")

    result = predict_all(route_id, day, phase)
    print(f"The predicted values are: Revenue: {result['Revenue']}, Expense: {result['Expense']}, Profit: {result['Profit']}")