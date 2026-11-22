def manage_peak(predicted_passengers):

    if predicted_passengers < 300000:
        level = "LOW"

        actions = [
            "Normal Operations",
            "Standard Staffing",
            "Regular Security Checkpoints"
        ]

    elif predicted_passengers < 700000:
        level = "MEDIUM"

        actions = [
            "Increase Check-in Counters",
            "Add Security Staff",
            "Prepare Extra Boarding Gates"
        ]

    else:
        level = "HIGH"

        actions = [
            "Open Additional Gates",
            "Increase Security Teams",
            "Increase Check-in Counters",
            "Deploy Passenger Guidance Staff",
            "Optimize Flight Scheduling",
            "Activate Peak Management Plan"
        ]

    return level, actions


predicted_passengers = 920000

level, actions = manage_peak(predicted_passengers)

print("Predicted Passengers:", predicted_passengers)
print("Peak Level:", level)

print("\nRecommended Actions:")

for action in actions:
    print("-", action)