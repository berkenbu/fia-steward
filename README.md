# FIA Steward Simulator

A Panda3D game where you are the race steward: watch a 3D race, spot incidents, and file penalty reports.

    pip install -r requirements.txt
    python fia_steward2.py

Graphics: PBR shading and 4096 shadow maps (simplepbr), textured track/grass, gradient sky dome.
Set `FIA_QUALITY=low` for the old fast renderer. To use a glTF car, put it at `models/car.glb`
(node `flap` is animated for DRS, nodes named `body*` get the team colour); otherwise the procedural car is used.

## Credits
`models/car.glb` is the f2002 open-wheeler from the low-poly F1 pack by som-natalino
(https://som-natalino.itch.io/low), CC0 1.0. Its texture carries real-world sponsor logos; the game recolours
the red livery per team (needs numpy, optional) but the logos remain. Swap in your own `models/car.glb` to change it.
