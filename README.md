# FIA Steward Simulator

A Panda3D game where you are the race steward: watch a 3D race, spot incidents, and file penalty reports.

    pip install -r requirements.txt
    python fia_steward2.py

Graphics: PBR shading and 4096 shadow maps (simplepbr), textured track/grass, gradient sky dome.
Set `FIA_QUALITY=low` for the old fast renderer. To use a glTF car, put it at `models/car.glb`
(node `flap` is animated for DRS, nodes named `body*` get the team colour); otherwise the procedural car is used.
