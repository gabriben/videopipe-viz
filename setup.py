from setuptools import setup, find_packages

setup(name='videopipeViz',
      version='0.1',
      description='visualization for Video Pipelines',
      url='http://github.com/gabriben/videopipe-viz',
      author='Cas Kok and Jordi Boon',
      author_email='',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'moviepy',
          'ffmpeg'
      ],
      zip_safe=False)
