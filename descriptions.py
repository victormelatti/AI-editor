# ✅ Manually restrict columns for the chatbot
custom_table_info = {
    "rnk_wur_2025_latest_vw": """This table contains the universities ranked in World University Ranking (WUR) 2025. 2025 refers to the year the ranking was published. The universities are ranked accross different subjects. If not otherwise specified, the overall rank must be used, i.e. subject = 'Overall'.
                               "Here the columns to be used and the meaning of each column: 
                                            subject: the subject of that university
                                            id: the univerisity ID,
                                            the_name: the name associated to the university ID,
                                            country: the country of the university,
                                            rank_number: the assigned rank number of that university in the subject,
                                            score_number: the score number of that university in the subject""",
    "rnk_wur_all_years_latest_vw": """ This table contains the universities ranked in World University Ranking (WUR) for all years, starting from 2016.
                                The universities are ranked accross different subjects. If not otherwise specified, the overall rank must be used, i.e. subject = 'Overall'.
                                Here the columns to be used and the meaning of each column: 
                                                                    wur_year: the year the ranking was published,
                                                                    subject: the subject of that university
                                                                    id: the univerisity ID,
                                                                    the_name: the name associated to the university ID,
                                                                    country: the country of the university,
                                                                    rank_number: the assigned rank number of that university in the subject,
                                                                    score_number: the score number of that university in the subject

    """
}

extract_parameters_from_query = {
                "name": "extract_parameters_from_query",
                "description": (
                    """ Extract parameters from a user query.
                        From the user query, extract the following:
                        - the metric of interest 'metric',
                        - a time period 'time_range' as a list of two years [start_year, end_year],
                        - a list of countries 'countries',
                        - what type of chart to generate 'chart_type' if the user has specified that a visualization is required,
                        - deduce what type of aggregation to apply ('aggregation') to generate a working SQL query that would answer the user query.
                        """
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": ["rank_number", "score_number", "id"],
                            "description": "The metric to plot. For 'count', the 'id' column (unique universities) is automatically used."
                        },
                        "time_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": """Time period for the ranking trend as a list of two years [start_year, end_year]. Current year is 2025, any reference to WUR followed by a number should be intended as a year, i.e WUR23 is 2023, WUR25 is 2025. 
                            If no time range is specified, the current year must be used is used."""
                        },
                        "countries": {
                            "type": "array", 
                            "items": {"type": "string"}, 
                            "description": "List of countries"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["line", "bar"],
                            "description": "Type of chart to generate as intended by the user"
                        },
                        "aggregation": {
                            "type": "string",
                            "enum": ["average", "count", "sum", "max", "min"],
                            "description": "Aggregation operation to apply (e.g., average, count, maximum, minimum)."
                        }
                    },
                   "required": ["time_range", "countries", "metric", "aggregation"]
                }
            }
        
