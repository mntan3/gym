def get_gripper_velocities(diagram, simulator, systems):
    sim_context = simulator.get_mutable_context()

    station_context = diagram.GetMutableSubsystemContext(
                          systems["station"], sim_context)
    velocities = []
    for arm_name in ["left", "right"]:
        state = systems["station"].get_model_state(station_context, arm_name)
        gripper_state = systems["station"].get_model_state(station_context, f"{arm_name}_gripper")
        velocities.extend(gripper_state["velocity"]*10)
        velocities.extend(state["velocity"])
    return velocities
