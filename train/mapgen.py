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
SPECIAL_DOOR_RAISE = 12       # Door_Raise(tag, speed, delay)
SPECIAL_DOOR_LOCKED = 13      # Door_LockedRaise(tag, speed, delay, lock)
SPECIAL_EXIT_NORMAL = 243     # Exit_Normal(pos)
LOCK_BLUE_CARD = 2            # Blue keycard / skull lock number in ZDoom


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
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        playeruse: bool = False,
        playercross: bool = False,
        repeatspecial: bool = False,
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
            ld["arg0"] = arg0
            if arg1 > 0:
                ld["arg1"] = arg1
            if arg2 > 0:
                ld["arg2"] = arg2
            if arg3 > 0:
                ld["arg3"] = arg3
        if playeruse:
            ld["playeruse"] = True
        if playercross:
            ld["playercross"] = True
        if repeatspecial:
            ld["repeatspecial"] = True
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
            for ak in ("arg0", "arg1", "arg2", "arg3"):
                if ak in l:
                    parts.append(f"{ak} = {l[ak]}")
            for fk in ("playeruse", "playercross", "repeatspecial"):
                if l.get(fk):
                    parts.append(f"{fk} = true")
            lines.append(f"linedef {{ {'; '.join(parts)}; }}\n")
        for t in self.things:
            parts = [
                f"x = {t['x']}", f"y = {t['y']}", f"angle = {t['angle']}", f"type = {t['type']}",
                "skill1 = true", "skill2 = true", "skill3 = true", "skill4 = true", "skill5 = true", "single = true"
            ]
            lines.append(f"thing {{ {'; '.join(parts)}; }}\n")
        return "".join(lines)


def generate_curriculum_map(stage: int = 1, seed: int = 42) -> MapBuilder:
    """Generates a solvable procedural UDMF map matching curriculum requirements.

    - Stage 1: Open corridor & rooms leading to GATE3 walkover exit pad
    - Stage 2: Combat arena with enemies & shotgun before exit pad
    - Stage 3: Closed manual door (special=12) funneling agent cleanly to Room 2 & exit
    - Stage 4: Blue Keycard in Room 1 -> Closed Blue Locked Door (special=13, arg3=2) -> Exit
    """
    rng = random.Random(seed)
    mb = MapBuilder()

    # Room 1: Player Start Room
    r1_x1, r1_y1 = 64.0, 128.0
    r1_w, r1_h = rng.randint(288, 416), rng.randint(288, 416)
    r1_x2, r1_y2 = r1_x1 + r1_w, r1_y1 + r1_h
    sec1 = mb.add_sector(floor_h=0, ceil_h=128, floor_tex="FLOOR0_1")

    # Door / Corridor dimensions (narrower on Stage 3/4 so agent squares up cleanly on door)
    has_door = stage in (3, 4)
    corridor_len = 64.0 if has_door else 96.0
    door_h = 96.0 if has_door else 112.0

    # Random vertical shift so the agent learns spatial steering rather than blind straight line
    max_shift = max(0, int((r1_h - door_h) / 2 - 48))
    y_shift = rng.randint(-max_shift, max_shift) if max_shift > 0 else 0
    door_y_bot = r1_y1 + (r1_h - door_h) / 2.0 + y_shift
    door_y_top = door_y_bot + door_h

    # Room 2: Goal Room
    r2_x1 = r1_x2 + corridor_len
    r2_h = 320.0
    r2_y1 = door_y_bot + (door_h - r2_h) / 2.0
    r2_w = rng.randint(288, 416)
    r2_x2, r2_y2 = r2_x1 + r2_w, r2_y1 + r2_h
    sec2 = mb.add_sector(floor_h=0, ceil_h=128, floor_tex="FLOOR4_8")

    # Door / Corridor Sector (ceil_h=0 when door starts closed!)
    sec_corr = mb.add_sector(
        floor_h=0,
        ceil_h=0 if has_door else 128,
        floor_tex="FLAT5_4" if has_door else "FLOOR0_3",
    )

    # Exit Pad Sector attached to East wall of Room 2
    exit_h = 96.0
    exit_shift = rng.randint(-64, 64)
    exit_y_bot = r2_y1 + (r2_h - exit_h) / 2.0 + exit_shift
    exit_y_top = exit_y_bot + exit_h
    sec_exit = mb.add_sector(floor_h=0, ceil_h=128, floor_tex="GATE3", light=255)

    # Vertices
    v0 = mb.add_vertex(r1_x1, r1_y1)
    v1 = mb.add_vertex(r1_x1, r1_y2)
    v2 = mb.add_vertex(r1_x2, r1_y2)
    v_p1_top = mb.add_vertex(r1_x2, door_y_top)
    v_p1_bot = mb.add_vertex(r1_x2, door_y_bot)
    v3 = mb.add_vertex(r1_x2, r1_y1)

    v_p2_top = mb.add_vertex(r2_x1, door_y_top)
    v_p2_bot = mb.add_vertex(r2_x1, door_y_bot)
    v5 = mb.add_vertex(r2_x1, r2_y2)
    v6 = mb.add_vertex(r2_x2, r2_y2)
    v_ex_top = mb.add_vertex(r2_x2, exit_y_top)
    v_ex_bot = mb.add_vertex(r2_x2, exit_y_bot)
    v7 = mb.add_vertex(r2_x2, r2_y1)
    v8 = mb.add_vertex(r2_x1, r2_y1)

    v_ex_tr = mb.add_vertex(r2_x2 + 64.0, exit_y_top)
    v_ex_br = mb.add_vertex(r2_x2 + 64.0, exit_y_bot)

    # Outer walls of Room 1 (clockwise -> front inside sec1)
    s_r1 = mb.add_sidedef(sec1, mid_tex="STARTAN2")
    mb.add_linedef(v0, v1, s_r1, blocking=True)
    mb.add_linedef(v1, v2, s_r1, blocking=True)
    mb.add_linedef(v2, v_p1_top, s_r1, blocking=True)
    mb.add_linedef(v_p1_bot, v3, s_r1, blocking=True)
    mb.add_linedef(v3, v0, s_r1, blocking=True)

    # Outer walls of Room 2 (clockwise -> front inside sec2)
    s_r2 = mb.add_sidedef(sec2, mid_tex="STONE2")
    mb.add_linedef(v_p2_top, v5, s_r2, blocking=True)
    mb.add_linedef(v5, v6, s_r2, blocking=True)
    mb.add_linedef(v6, v_ex_top, s_r2, blocking=True)
    mb.add_linedef(v_ex_bot, v7, s_r2, blocking=True)
    mb.add_linedef(v7, v8, s_r2, blocking=True)
    mb.add_linedef(v8, v_p2_bot, s_r2, blocking=True)

    # Corridor / Door side walls (front inside sec_corr)
    side_tex = "DOORTRAK" if has_door else "STARTAN2"
    s_corr = mb.add_sidedef(sec_corr, mid_tex=side_tex)
    mb.add_linedef(v_p1_top, v_p2_top, s_corr, blocking=True)
    mb.add_linedef(v_p2_bot, v_p1_bot, s_corr, blocking=True)

    # Portals: Room1 <-> Door/Corr, Door/Corr <-> Room2
    door_upper = "BIGDOOR2" if stage == 3 else ("BIGDOOR4" if stage >= 4 else "-")
    s_r1_portal = mb.add_sidedef(sec1, mid_tex="-", top_tex=door_upper)
    s_corr_p1 = mb.add_sidedef(sec_corr, mid_tex="-")
    s_r2_portal = mb.add_sidedef(sec2, mid_tex="-", top_tex=door_upper)
    s_corr_p2 = mb.add_sidedef(sec_corr, mid_tex="-")

    if stage == 3:
        mb.add_linedef(
            v_p1_top, v_p1_bot, s_r1_portal, s_corr_p1, blocking=False,
            special=SPECIAL_DOOR_RAISE, arg0=0, arg1=16, arg2=150,
            playeruse=True, repeatspecial=True,
        )
        mb.add_linedef(
            v_p2_bot, v_p2_top, s_r2_portal, s_corr_p2, blocking=False,
            special=SPECIAL_DOOR_RAISE, arg0=0, arg1=16, arg2=150,
            playeruse=True, repeatspecial=True,
        )
    elif stage >= 4:
        mb.add_linedef(
            v_p1_top, v_p1_bot, s_r1_portal, s_corr_p1, blocking=False,
            special=SPECIAL_DOOR_LOCKED, arg0=0, arg1=16, arg2=150, arg3=LOCK_BLUE_CARD,
            playeruse=True, repeatspecial=True,
        )
        mb.add_linedef(
            v_p2_bot, v_p2_top, s_r2_portal, s_corr_p2, blocking=False,
            special=SPECIAL_DOOR_LOCKED, arg0=0, arg1=16, arg2=150, arg3=LOCK_BLUE_CARD,
            playeruse=True, repeatspecial=True,
        )
    else:
        mb.add_linedef(v_p1_top, v_p1_bot, s_r1_portal, s_corr_p1, blocking=False)
        mb.add_linedef(v_p2_bot, v_p2_top, s_r2_portal, s_corr_p2, blocking=False)

    # Exit Pad Walkover Portal (Room 2 <-> sec_exit, special 243 playercross=true)
    s_r2_exit = mb.add_sidedef(sec2, mid_tex="-")
    s_exit_portal = mb.add_sidedef(sec_exit, mid_tex="-")
    mb.add_linedef(
        v_ex_top, v_ex_bot, s_r2_exit, s_exit_portal, blocking=False,
        special=SPECIAL_EXIT_NORMAL, playercross=True,
    )

    # Exit Pad outer alcove walls (front inside sec_exit)
    s_exit_wall = mb.add_sidedef(sec_exit, mid_tex="GRAY1")
    mb.add_linedef(v_ex_top, v_ex_tr, s_exit_wall, blocking=True)
    mb.add_linedef(v_ex_tr, v_ex_br, s_exit_wall, blocking=True)
    mb.add_linedef(v_ex_br, v_ex_bot, s_exit_wall, blocking=True)

    # Player 1 Start in Room 1 (with slightly randomized initial angle toward East)
    start_ang = rng.choice([0, 15, 345, 30, 330])
    mb.add_thing(r1_x1 + 80.0, r1_y1 + r1_h / 2.0, THING_PLAYER_START, angle=start_ang)

    # Stage-specific things
    if stage == 1:
        mb.add_thing(r2_x2 - 96.0, exit_y_bot + exit_h / 2.0, THING_STIMPACK)
    elif stage == 2:
        mb.add_thing(r2_x1 + r2_w * 0.55, r2_y1 + r2_h * 0.5, THING_ZOMBIEMAN, angle=180)
        mb.add_thing(r2_x2 - 96.0, r2_y1 + 96.0, THING_IMP, angle=180)
        mb.add_thing(r1_x1 + 144.0, r1_y1 + 80.0, THING_SHOTGUN)
        mb.add_thing(r1_x1 + 144.0, r1_y2 - 80.0, THING_AMMO_SHELLS)
    elif stage == 3:
        mb.add_thing(r2_x1 + r2_w * 0.55, r2_y1 + r2_h * 0.5, THING_ZOMBIEMAN, angle=180)
        mb.add_thing(r2_x2 - 80.0, r2_y1 + 80.0, THING_MEDIKIT)
    elif stage >= 4:
        # Blue keycard in Room 1 near upper/lower corner so agent collects it before locked door
        key_y = r1_y2 - 72.0 if (seed % 2 == 0) else r1_y1 + 72.0
        mb.add_thing(r1_x1 + r1_w * 0.55, key_y, THING_BLUE_CARD)
        mb.add_thing(r2_x1 + r2_w * 0.55, r2_y1 + r2_h * 0.5, THING_ZOMBIEMAN, angle=180)
        mb.add_thing(r2_x2 - 80.0, r2_y1 + 80.0, THING_MEDIKIT)

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
