.. pyhydrosym documentation master file, created by
   sphinx-quickstart on Thu Feb 29 18:55:37 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

   tutorials/tutorials
   api_reference
   credits
   changelog
   known_issues   


Добро пожаловать в документацию фреймворка pyhydrosym!
======================================================

Pyhydrosim -- фреймворк для суррогатного моделирования и оптимизации трубопроводных систем нефтегазовой отрасли. 
Фреймворк разработан на базе :pytorch:`null` `PyTorch <https://pytorch.org>`_ . 
*Pyhydrosim* содержит коннекторы к газо- и гидродинамическим симуляторам, механизмы для создания суррогатных моделей различных видов 
(на основе методов машинного обучения, гаусовских процессов, глубоких нейронных сетей, включая нейронные операторы и графовые нейронные сети),
включая средства планирования и управления вычислительными экспериментами, а также интерфейсы взаимодействия с различными оптимизаторами - 
солверами для решения задач нелинейного программирования. Фреймворк содержит инструменты для моделирования больших систем ("систем систем")
на основе глубокого машинного обучения на графах, известного как `геометрическое машинное обучение <http://geometricdeeplearning.com/>`_.
*Pyhydrosim* включает простые в использовании мини-бэтч загрузчики, поддержки режима работы в кластере GPU,
поддержку `torch.compile <https://pytorch-geometric.readthedocs.io/en/latest/advanced/compile.html>`_, бенчмарк-датасеты, "гимнастический зал" моделей,
удобные трансформеры данных. Фреймворк использует `Pytorch Lightning <https://pytorch-lightning.readthedocs.io>`_, который предназначен для обучения на CPUs, отдельных и кластерных GPUs.

.. toctree::
  :glob:
  :maxdepth: 2
  :caption: Содержание:

  notes/install
  notes/overview
   
.. toctree::
  :glob:
  :maxdepth: 3
  :caption: Модули:

  modules/modules