from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            # Include the first Launch file
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("fast_livo"), "launch", "online_relo.launch.py"]
                        )
                    ]
                ),
                # Arguments can be passed to the included Launch file
                # launch_arguments={
                #    'arg_name': 'value',
                #    'another_arg': 'value2'
                # }.items()
            ),
            # Include the second Launch file
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("fast_livo"), "launch", "mapping_g1.launch.py"]
                        )
                    ]
                )
            ),
            # Include the third Launch file
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("g1_navigation2"), "launch", "navigation2.launch.py"]
                        )
                    ]
                )
            ),
            # Start standalone node 1
            # Node(
            #    package='tfpub',
            #    executable='tfpub',
            #    name='tfpub',
            #    namespace='tfpub',
            #    #parameters=[{'param_name': 'param_value'}],
            #    #remappings=[('topic1', 'topic2')]
            # )
        ]
    )
