.. role:: raw-html-m2r(raw)
   :format: html


HyPyP 🐍〰️🐍
=============

The **Hy**\ perscanning **Py**\ thon **P**\ ipeline


.. image:: https://img.shields.io/pypi/v/hypyp.svg
   :target: https://pypi.org/project/HyPyP/
   :alt: PyPI version shields.io
 :raw-html-m2r:`<a href="https://travis-ci.org/ppsp-team/HyPyP"><img src="https://travis-ci.org/ppsp-team/HyPyP.svg?branch=master"></a>` 
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: license
 
.. image:: https://img.shields.io/static/v1?label=chat&message=Mattermost&color=Blue
   :target: https://mattermost.brainhack.org/brainhack/channels/hypyp
   :alt: Mattermost


:warning: This alpha version is still far from easy-to-use and should be considered with caution. While we have done our best to test all the functionalities, there is no guarantee that the pipeline is entirely bug-free. 

📖 See our `preprint <https://psyarxiv.com/x5apu>`_ for more explanation and our plan for upcoming functionalities (aka Roadmap).

🤝 If you want to help you can submit bugs and suggestions of enhancements in our Github `Issues section <https://github.com/ppsp-team/HyPyP/issues>`_.

🤓 For the motivated contributors, you can even help directly in the developpment of HyPyP. You will need to install `Poetry <https://python-poetry.org/>`_ (see section below).

Contributors
------------

Florence BRUN, Anaël AYROLLES, Phoebe CHEN, Amir DJALOVSKI, Yann BEAUXIS, Suzanne DIKKER, Guillaume DUMAS

Installation
------------

.. code-block::

   pip install HyPyP

Documentation
-------------

HyPyP documentation of all the API functions is available online at `hypyp.readthedocs.io <https://hypyp.readthedocs.io/>`_

For getting started with HyPyP, we have designed a little walkthrough: `getting_started.ipynb <https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb>`_

API
---

🛠 `io.py <https://github.com/ppsp-team/HyPyP/blob/master/hypyp/io.py>`_ — Loaders (Florence, Anaël, Guillaume)

🧰 `utils.py <https://github.com/ppsp-team/HyPyP/blob/master/hypyp/utils.py>`_ — Basic tools (Amir, Florence, Guilaume)

⚙️ `prep.py <https://github.com/ppsp-team/HyPyP/blob/master/hypyp/prep.py>`_ — Preprocessing (ICA & AutoReject) (Anaël, Florence, Guillaume)

🔠 `analyses.py <https://github.com/ppsp-team/HyPyP/blob/master/hypyp/analyses.py>`_ — Power spectral density and wide choice of connectivity measures (Phoebe, Suzanne, Florence, Guillaume)

📈 `stats.py <https://github.com/ppsp-team/HyPyP/blob/master/hypyp/stats.py>`_ — Statistics (permutations & cluster statistics) (Florence, Guillaume)

🧠 `viz.py <https://github.com/ppsp-team/HyPyP/blob/master/hypyp/viz.py>`_ — Inter-brain visualization (Anaël, Amir, Florence, Guillaume)

🎓 `Tutorials <https://github.com/ppsp-team/HyPyP/tree/master/tutorial>`_ - Examples & documentation (Anaël, Florence, Yann, Guillaume)

Poetry installation (only for developpers)
------------------------------------------

Step 1: ``pip install poetry``

Step 2: ``git clone git@github.com:ppsp-team/HyPyP.git``

Step 3: ``cd HyPyP``

Step 4: ``poetry install``

Step 5: ``poetry shell``
