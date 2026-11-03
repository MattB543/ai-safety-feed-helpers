#!/usr/bin/env python3
"""Compare AISafetyFeed following list vs leaderboard."""

following = [
    'AISafetyMemes','ARGleave','AlecStapp','Altimor','AmandaAskell',
    'AndrewCritchPhD','AndrewCurran_','AnthropicAI','ArthurB','BethMayBarnes',
    'CAIS','CharlesMonneron','DavidSKrueger','Dorialexander','ESYudkowsky',
    'EpochAIResearch','EvanHub','GarrisonLovely','GoogleDeepMind','JacobSteinhardt',
    'JeffLadish','Kat__Woods','KatjaGrace','KelseyTuoc','MariusHobbhahn',
    'MatthewJBar','MichaelTrazzi','Miles_Brundage','NPCollapse','NateSilver538',
    'NathanpmYoung','NeelNanda5','NunoSempere','OfficialLoganK','Ollie_Base',
    'OpenAI','OwainEvans_UK','QuintinPope5','RichardMCNgo','S_OhEigeartaigh',
    'SethBurn','ShakeelHashim','Simeon_Cps','So8res','StefanFSchubert',
    'TheZvi','Thomas_Woodside','Turn_Trout','aidan_mclau','ajeya_cotra',
    'albrgr','alexeyguzey','apolloaievals','atroyn','ben_j_todd',
    'benlandautaylor','binarybits','connoraxiotes','danfaggella','daniel_271828',
    'davidmanheim','deanwball','deedydas','dfrsrchtwts','dhadfieldmenell',
    'dwarkesh_sp','emollick','eshear','gallabytes','gdb',
    'goodside','hendrycks','hlntnr','jachiam0','jackclarkSF',
    'jam3scampbell','janleike','jd_pressman','jjding99','jkcarlsmith',
    'johnschulman2','karpathy','krishnanrohit','leopoldasch','lilianweng',
    'liron','lukeprog','lxrjl','nabeelqu','nabla_theta',
    'nearcyan','norabelrose','ohabryka','ohlennart','oscredwin',
    'ozziegooen','peterwildeford','polynoamial','r_zwetsloot','rao2z',
    'repligate','robbensinger','robertskmiles','robertwiblin','robinhanson',
    'rohinmshah','s_r_constantin','sebkrier','slatestarcodex','sleepinyourhat',
    'stanislavfort','tamaybes','tegmark','teortaxesTex','tobyordoxford',
    'tsarnick','tszzl','tyler_m_john','tylercowen','willmacaskill',
]

# Leaderboard list (extract usernames from URLs)
leaderboard = []
with open('20260304_final_leaderboard_300_x_links.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            username = line.rstrip('/').split('/')[-1]
            leaderboard.append(username)

# Case-insensitive comparison
following_lower = {u.lower() for u in following}
leaderboard_lower = {u.lower() for u in leaderboard}
leaderboard_map = {u.lower(): u for u in leaderboard}
following_map = {u.lower(): u for u in following}

overlap = following_lower & leaderboard_lower
only_following = following_lower - leaderboard_lower
only_leaderboard = leaderboard_lower - following_lower

print(f'Following: {len(following)}')
print(f'Leaderboard: {len(leaderboard)}')
print(f'Overlap (on both): {len(overlap)}')
print(f'Only in following (not on leaderboard): {len(only_following)}')
print(f'Only on leaderboard (not following): {len(only_leaderboard)}')

print(f'\n=== On both ({len(overlap)}) ===')
for u in sorted(overlap):
    print(f'  @{following_map.get(u, u)}')

print(f'\n=== Following but NOT on leaderboard ({len(only_following)}) ===')
for u in sorted(only_following):
    print(f'  @{following_map.get(u, u)}')

print(f'\n=== On leaderboard but NOT following ({len(only_leaderboard)}) ===')
for u in sorted(only_leaderboard):
    print(f'  @{leaderboard_map.get(u, u)}')
