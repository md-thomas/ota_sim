from .channel import OTALink, free_space_path_loss_db, propagation_delay_s

__all__ = ['OTALink', 'free_space_path_loss_db', 'propagation_delay_s']

# Deliberately NOT importing from .simulator here: simulator.py is meant to
# also be run directly via `python3 -m ota_sim.simulator`, and eagerly
# importing it into this __init__ would make Python load it twice (once via
# this package import, once as __main__), triggering a RuntimeWarning.
# Import OTASimulator/encode_peer_id/decode_peer_id/PEER_ID_LEN from
# ota_sim.simulator directly if you need them as a library.
