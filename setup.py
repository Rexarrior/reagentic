from setuptools import setup, find_packages

setup(
    name='reagentic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai-agents',
        'pydantic>=2.0.0',
        'pytest-asyncio>=1.0.0',
    ],
    author='Ksandr Renderon',
    author_email='rexarrior@example.com',
    description='Tool library to create agents with openai-agents',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rexarrior/reagentic',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.13',
)
