from mcs.main import JusticeLeague

if __name__ == "__main__":
    # Example patient case
    patient_case = """
    Patient: 45-year-old White Male
    Location: New York, NY

    Lab Results:
    - egfr 
    - 59 ml / min / 1.73
    - non african-american
    
    """

    swarm = JusticeLeague(
        patient_id="323u29382938293829382382398",
        max_loops=1,
        output_type="json",
        patient_documentation="",
    )

    swarm.run(task=patient_case)

    # print(json.dumps(swarm.to_dict()))
