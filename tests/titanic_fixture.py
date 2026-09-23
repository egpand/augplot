"""Illustrative timeline events transcribed from the supplied notebook."""

import pandas as pd


def titanic_timeline_events() -> pd.DataFrame:
    """Provide the events independently of the Kaggle passenger CSV."""
    return pd.DataFrame(
        {
            "x": [1.5, 2.4, 2.9, 3.4, 3.8, 4.5, 5.0, 6.5, 7.0, 7.6, 8.0],
            "time": [
                "1:30 PM",
                "9:00 AM",
                "1:42 PM",
                "7:15 PM",
                "10:00 PM",
                "11:30 PM",
                "11:40 PM",
                "12:20 AM",
                "12:45 AM",
                "2:00 AM",
                "2:20 AM",
            ],
            "event": [
                "Titanic sets sail",
                "Message received",
                "Baltic warns of\nicebergs",
                "Smith requests\nreturn of message",
                "Lightoller relieved\nfrom duty",
                "Iceberg warning\nbells",
                "Titanic hits\nan iceberg",
                "Lifeboats lowered",
                "Passengers slowly\narrive on deck",
                "Stern begins\nto rise",
                "Titanic sinks",
            ],
            "side": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
            "highlight": [False] * 10 + [True],
        }
    )
