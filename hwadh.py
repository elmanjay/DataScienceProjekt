dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Auswahl der Datengrundlage:", className="font-weight-bold"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.RadioItems(
                                            id="zeitraum-arima",
                                            options=[
                                                {"label": "1 Jahr", "value": 1},
                                                {"label": "2 Jahre (empfohlen)", "value": 2},
                                                {"label": "5 Jahre", "value": 5},
                                            ],
                                            value=2,
                                            className="radiobuttons",
                                            labelStyle={"display": "inline-block", "margin-right": "5px"},
                                            inline=True,
                                        ),
                                    ),
                                    dbc.Col(
                                        [
                                        dbc.Label("Auswahl der Parameter:", className="font-weight-bold"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("P:", className="font-weight-bold"),
                                                        ],
                                                        width="auto",
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            type="text",
                                                            inputMode="numeric",
                                                            pattern="[0-9]*",
                                                            placeholder="Ganzzahl eingeben",
                                                            className="small-input",
                                                        ),
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("D:", className="font-weight-bold"),
                                                        ],
                                                        width="1",
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            type="text",
                                                            inputMode="numeric",
                                                            pattern="[0-9]*",
                                                            placeholder="Ganzzahl eingeben",
                                                            className="small-input",
                                                        ),
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Q:", className="font-weight-bold"),
                                                        ],
                                                        width="1",
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            type="text",
                                                            inputMode="numeric",
                                                            pattern="[0-9]*",
                                                            placeholder="Ganzzahl eingeben",
                                                            className="small-input",
                                                        ),
                                                    ),
                                                ],
                                                align="center",
                                                className="mb-2",
                                            ),
                                        ],
                                    ),
                                ],
                                align="center",
                            ),
                        ],
                    ),
                ],
                justify="center",
            ),
        ],
        fluid=True,
    ),