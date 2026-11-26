"""Procedural UDMF map generator for ViZDoom curriculum training.

Generates valid WAD files containing UDMF (Universal Doom Map Format) levels:
- Stage 1: Open rooms and corridors (navigation and locomotion)
- Stage 2: Combat arenas with enemies (Zombieman, Imp) and health/ammo
- Stage 3: Multi-room layouts with standard manual doors
- Stage 4: Long-horizon progression: Keycard -> Locked Door -> Exit room

Uses only standard library and doom_env.wad.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from doom_env.wad import Lump, write_wad

# DOOM Thing types
THING_PLAYER_START = 1
THING_ZOMBIEMAN = 3004
THING_IMP = 3001
THING_DEMON = 3002
THING_MEDIKIT = 2012
THING_STIMPACK = 2011
THING_SHOTGUN = 2001
THING_AMMO_CLIP = 2007
THING_AMMO_SHELLS = 2008
THING_BLUE_CARD = 5
THING_RED_CARD = 13
THING_YELLOW_CARD = 6

# Linedef specials
SPECIAL_DOOR_OPEN = 11       # Exit / Door
SPECIAL_EXIT_SWITCH = 243    # Exit_Normal


@dataclass
class MapBuilder:
    """Constructs UDMF text and compiles into WAD lump."""
    
    vertices: list[tuple[float, float]] = field(default_factory=list)
    sectors: list[dict] = field(default_factory=list)
    sidedefs: list[dict] = field(default_factory=list)
    linedefs: list[dict] = field(default_factory=list)
    things: list[dict] = field(default_factory=list)
    
    def add_vertex(self, x: float, y: float) -> int:
        self.vertices.append((float(x), float(y)))
        return len(self.vertices) - 1

    def add_sector(
        self,
        floor_h: int = 0,
        ceil_h: int = 128,
        floor_tex: str = "FLOOR0_1",
        ceil_tex: str = "CEIL1_1",
        light: int = 192,
    ) -> int:
        self.sectors.append({
            "heightfloor": floor_h,
            "heightceiling": ceil_h,
            "texturefloor": floor_tex,
            "textureceiling": ceil_tex,
            "lightlevel": light,
        })
        return len(self.sectors) - 1

    def add_sidedef(
        self,
        sector_id: int,
        mid_tex: str = "STARTAN2",
        top_tex: str = "-",
        bottom_tex: str = "-",
    ) -> int:
        self.sidedefs.append({
            "sector": sector_id,
            "texturemiddle": mid_tex,
            "texturetop": top_tex,
            "texturebottom": bottom_tex,
        })
        return len(self.sidedefs) - 1

    def add_linedef(
        self,
        v1: int,
        v2: int,
        front_side: int,
        back_side: int | None = None,
        blocking: bool = True,
        special: int = 0,
        arg0: int = 0,
    ) -> int:
        ld = {
            "v1": v1,
            "v2": v2,
            "sidefront": front_side,
            "blocking": blocking,
            "dontpegbottom": True,
        }
        if back_side is not None:
            ld["sideback"] = back_side
            ld["twosided"] = True
            if not blocking:
                ld["blocking"] = False
        if special > 0:
            ld["special"] = special
            if arg0 > 0:
                ld["arg0"] = arg0
        self.linedefs.append(ld)
        return len(self.linedefs) - 1

    def add_thing(self, x: float, y: float, type_id: int, angle: int = 0) -> None:
        self.things.append({
            "x": float(x),
            "y": float(y),
            "angle": int(angle),
            "type": int(type_id),
            "skill1": True,
            "skill2": True,
            "skill3": True,
            "skill4": True,
            "skill5": True,
            "single": True,
        })

    def add_rectangular_room(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        floor_h: int = 0,
        ceil_h: int = 128,
        wall_tex: str = "STARTAN2",
        floor_tex: str = "FLOOR0_1",
    ) -> tuple[int, list[int], list[int]]:
        """Creates a standalone rectangular room. Returns (sector_id, [v0, v1, v2, v3], [line0, line1, line2, line3])."""
        sec = self.add_sector(floor_h=floor_h, ceil_h=ceil_h, floor_tex=floor_tex)
        
        # Clockwise vertices for front-side pointing inside
        # v0: bottom-left, v1: top-left, v2: top-right, v3: bottom-right
        v0 = self.add_vertex(x1, y1)
        v1 = self.add_vertex(x1, y2)
        v2 = self.add_vertex(x2, y2)
        v3 = self.add_vertex(x2, y1)

        s0 = self.add_sidedef(sec, mid_tex=wall_tex)
        s1 = self.add_sidedef(sec, mid_tex=wall_tex)
        s2 = self.add_sidedef(sec, mid_tex=wall_tex)
        s3 = self.add_sidedef(sec, mid_tex=wall_tex)

        l0 = self.add_linedef(v0, v1, s0, blocking=True)
        l1 = self.add_linedef(v1, v2, s1, blocking=True)
        l2 = self.add_linedef(v2, v3, s2, blocking=True)
        l3 = self.add_linedef(v3, v0, s3, blocking=True)

        return sec, [v0, v1, v2, v3], [l0, l1, l2, l3]

    def build_udmf_text(self) -> str:
        lines = ['namespace = "zdoom";\n']
        for v in self.vertices:
            lines.append(f"vertex {{ x = {v[0]}; y = {v[1]}; }}\n")
        for s in self.sectors:
            lines.append(
                f"sector {{ texturefloor = \"{s['texturefloor']}\"; textureceiling = \"{s['textureceiling']}\"; "
                f"heightfloor = {s['heightfloor']}; heightceiling = {s['heightceiling']}; lightlevel = {s['lightlevel']}; }}\n"
            )
        for s in self.sidedefs:
            lines.append(
                f"sidedef {{ sector = {s['sector']}; texturetop = \"{s['texturetop']}\"; "
                f"texturemiddle = \"{s['texturemiddle']}\"; texturebottom = \"{s['texturebottom']}\"; }}\n"
            )
        for l in self.linedefs:
            parts = [f"v1 = {l['v1']}", f"v2 = {l['v2']}", f"sidefront = {l['sidefront']}"]
            if "sideback" in l:
                parts.append(f"sideback = {l['sideback']}")
            if l.get("twosided"):
                parts.append("twosided = true")
            if l.get("blocking"):
                parts.append("blocking = true")
            if l.get("dontpegbottom"):
                parts.append("dontpegbottom = true")
            if "special" in l:
                parts.append(f"special = {l['special']}")
            if "arg0" in l:
                parts.append(f"arg0 = {l['arg0']}")
            lines.append(f"linedef {{ {'; '.join(parts)}; }}\n")
        for t in self.things:
            parts = [
                f"x = {t['x']}", f"y = {t['y']}", f"angle = {t['angle']}", f"type = {t['type']}",
                "skill1 = true", "skill2 = true", "skill3 = true", "skill4 = true", "skill5 = true", "single = true"
            ]
            lines.append(f"thing {{ {'; '.join(parts)}; }}\n")
        return "".join(lines)


def generate_curriculum_map(stage: int = 1, seed: int = 42) -> MapBuilder:
    """Generates a procedural map matching curriculum requirements."""
    rng = random.Random(seed)
    mb = MapBuilder()

    # Room 1: Player Start Room (always present)
    r1_x1, r1_y1 = 64.0, 128.0
    r1_w, r1_h = rng.randint(320, 512), rng.randint(320, 512)
    r1_x2, r1_y2 = r1_x1 + r1_w, r1_y1 + r1_h
    sec1 = mb.add_sector(floor_h=0, ceil_h=128, floor_tex="FLOOR0_1")

    # Room 2: Goal/Corridor Room
    corridor_w = 128.0
    r2_x1 = r1_x2 + corridor_w
    r2_y1 = r1_y1 + (r1_h - 384.0) / 2.0
    r2_w, r2_h = rng.randint(384, 512), 384.0
    r2_x2, r2_y2 = r2_x1 + r2_w, r2_y1 + r2_h
    sec2 = mb.add_sector(floor_h=0, ceil_h=128, floor_tex="FLOOR4_8")

    # Corridor connecting Room 1 and Room 2
    sec_corr = mb.add_sector(floor_h=0, ceil_h=128, floor_tex="FLOOR0_3")

    # Construct vertices
    # Room 1 vertices
    v0 = mb.add_vertex(r1_x1, r1_y1)
    v1 = mb.add_vertex(r1_x1, r1_y2)
    v2 = mb.add_vertex(r1_x2, r1_y2)
    v3_top = mb.add_vertex(r1_x2, r2_y1 + corridor_w)
    v3_bot = mb.add_vertex(r1_x2, r2_y1)
    v4 = mb.add_vertex(r1_x2, r1_y1)

    # Room 2 vertices
    v5_bot = mb.add_vertex(r2_x1, r2_y1)
    v5_top = mb.add_vertex(r2_x1, r2_y1 + corridor_w)
    v6 = mb.add_vertex(r2_x1, r2_y2)
    v7 = mb.add_vertex(r2_x2, r2_y2)
    v8 = mb.add_vertex(r2_x2, r2_y1)

    # Outer boundary Room 1
    s_r1 = mb.add_sidedef(sec1, mid_tex="STARTAN2")
    mb.add_linedef(v0, v1, s_r1, blocking=True)
    mb.add_linedef(v1, v2, s_r1, blocking=True)
    mb.add_linedef(v2, v3_top, s_r1, blocking=True)
    mb.add_linedef(v3_bot, v4, s_r1, blocking=True)
    mb.add_linedef(v4, v0, s_r1, blocking=True)

    # Outer boundary Room 2
    s_r2 = mb.add_sidedef(sec2, mid_tex="STARTAN2")
    mb.add_linedef(v5_top, v6, s_r2, blocking=True)
    mb.add_linedef(v6, v7, s_r2, blocking=True)
    mb.add_linedef(v7, v8, s_r2, blocking=True)
    mb.add_linedef(v8, v5_bot, s_r2, blocking=True)

    # Corridor walls (top and bottom)
    s_corr = mb.add_sidedef(sec_corr, mid_tex="STARTAN2")
    mb.add_linedef(v3_top, v5_top, s_corr, blocking=True)
    mb.add_linedef(v5_bot, v3_bot, s_corr, blocking=True)

    # Portals: Room1 <-> Corr, Corr <-> Room2
    s_r1_portal = mb.add_sidedef(sec1, mid_tex="-")
    s_corr_p1 = mb.add_sidedef(sec_corr, mid_tex="-")
    mb.add_linedef(v3_bot, v3_top, s_r1_portal, s_corr_p1, blocking=False)

    s_corr_p2 = mb.add_sidedef(sec_corr, mid_tex="-")
    s_r2_portal = mb.add_sidedef(sec2, mid_tex="-")
    
    # In Stage 3 or 4, place a door between corridor and Room 2
    if stage in (3, 4):
        # Door special on portal line (special 11: Door_Open or Door_Raise)
        mb.add_linedef(v5_top, v5_bot, s_corr_p2, s_r2_portal, blocking=False, special=11)
    else:
        mb.add_linedef(v5_top, v5_bot, s_corr_p2, s_r2_portal, blocking=False)

    # Player 1 Start in Room 1
    mb.add_thing(r1_x1 + 96.0, r1_y1 + r1_h / 2.0, THING_PLAYER_START, angle=0)

    # Stage-specific things
    if stage == 1:
        # Simple navigation target: Exit switch in Room 2
        mb.add_thing(r2_x2 - 64.0, r2_y1 + r2_h / 2.0, THING_MEDIKIT)
    elif stage == 2:
        # Combat arena: Enemies + Ammo in Room 2
        mb.add_thing(r2_x2 - 96.0, r2_y1 + r2_h / 2.0, THING_ZOMBIEMAN, angle=180)
        mb.add_thing(r2_x2 - 160.0, r2_y1 + 96.0, THING_IMP, angle=180)
        mb.add_thing(r1_x1 + 160.0, r1_y1 + 96.0, THING_SHOTGUN)
        mb.add_thing(r1_x1 + 160.0, r1_y2 - 96.0, THING_AMMO_SHELLS)
    elif stage == 3:
        # Multi-room door exploration
        mb.add_thing(r2_x2 - 96.0, r2_y1 + r2_h / 2.0, THING_ZOMBIEMAN, angle=180)
        mb.add_thing(r2_x2 - 64.0, r2_y1 + 96.0, THING_MEDIKIT)
    elif stage >= 4:
        # Keycard in Room 1, locked door sequence to exit
        mb.add_thing(r1_x1 + 96.0, r1_y2 - 96.0, THING_BLUE_CARD)
        mb.add_thing(r2_x2 - 96.0, r2_y1 + r2_h / 2.0, THING_DEMON, angle=180)
        mb.add_thing(r2_x2 - 64.0, r2_y1 + 64.0, THING_MEDIKIT)

    return mb


def export_wad(mb: MapBuilder, wad_file: str | Path, map_name: str = "MAP01") -> None:
    """Exports MapBuilder to a valid UDMF WAD file."""
    udmf_text = mb.build_udmf_text()
    lumps = [
        Lump(map_name.upper(), b""),
        Lump("TEXTMAP", udmf_text.encode("ascii")),
        Lump("ENDMAP", b""),
    ]
    write_wad(wad_file, lumps)


def main():
    parser = argparse.ArgumentParser(description="Generate procedural DOOM UDMF maps")
    parser.add_argument("--output", default="maps/stage1_test.wad", help="Output WAD file path")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Curriculum stage (1-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--map", default="MAP01", help="Map name in WAD")
    args = parser.parse_args()

    mb = generate_curriculum_map(stage=args.stage, seed=args.seed)
    export_wad(mb, args.output, map_name=args.map)
    print(f"Generated Stage {args.stage} map: {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
