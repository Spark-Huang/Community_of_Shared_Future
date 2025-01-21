from cosf.main import CommunityOfSharedFuture

if __name__ == "__main__":
    # Extended Example Patient Case
    criminals_case = """
    Criminals Information:
    - Name: John Doe
    - Age: 45
    - Gender: Male
    - Ethnicity: White
    - Location: New York, NY
    - BMI: 28.5 (Overweight)
    - Occupation: Office Worker

    Facts of the case:
    - Defrauding the AI digital person Luna   
    """

    # Initialize the CommunityOfSharedFuture with the detailed patient case
    swarm = CommunityOfSharedFuture(
        patient_id="Patient-001",
        max_loops=1,
    )

    # Run the swarm on the patient case
    output = swarm.run(task=criminals_case)

    # Print the system's state after processing
    print(output.model_dump_json(indent=4))