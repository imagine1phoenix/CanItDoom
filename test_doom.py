import os
import vizdoom as vd

g = vd.DoomGame()
doom_path = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
g.set_doom_game_path(doom_path)
g.set_doom_map("MAP01")
g.set_mode(vd.Mode.ASYNC_SPECTATOR)
g.set_window_visible(True)
g.set_sound_enabled(True)
g.set_screen_resolution(vd.ScreenResolution.RES_640X480)
g.set_available_buttons([
    vd.Button.MOVE_FORWARD,
    vd.Button.MOVE_BACKWARD,
    vd.Button.MOVE_LEFT,
    vd.Button.MOVE_RIGHT,
    vd.Button.TURN_LEFT,
    vd.Button.TURN_RIGHT,
    vd.Button.TURN_LEFT_RIGHT_DELTA,
    vd.Button.ATTACK,
    vd.Button.USE,
    vd.Button.SELECT_NEXT_WEAPON,
    vd.Button.SPEED,
])
g.init()
g.new_episode()
print("DOOM window open (WASD/Arrows move, Mouse/Arrows turn, Ctrl/Click shoot, E/Space use).", flush=True)

try:
    while not g.is_episode_finished():
        g.advance_action()
except (vd.ViZDoomUnexpectedExitException, KeyboardInterrupt):
    print("DOOM window closed.")
finally:
    try:
        g.close()
    except Exception:
        pass

