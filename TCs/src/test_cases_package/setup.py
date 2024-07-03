from setuptools import find_packages, setup
import os

package_name = 'test_cases_package'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/webots_launch_tc1.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/webots_launch_tc2.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/webots_launch_tc3.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/test_world.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/test_world_2.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/test_world_3.wbt']))
data_files.append(('share/' + package_name + '/resource', ['resource/E-puck.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='javi2002bj',
    maintainer_email='javier.ballesteros@uclm.es',
    description='Package of the Test Case 1 of the Project',
    license='Apache2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = test_cases_package.robot_controller:main'
        ],
    },
)
