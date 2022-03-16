# Lava spread https://i.imgur.com/WC5urIh.png (3/16^3 chance every tick)
# Fire spread https://i.imgur.com/gt6Ari4.png (every 30-39 ticks, uniformly randomly)

from nbt import nbt
import os
import numpy as np
from numpy import random
from numpy.lib.function_base import percentile
import matplotlib.pyplot as plt
import json
import scipy.stats 

designs_path = "flintless\\designs"
design_files = [f for f in os.listdir(designs_path)]

FIRE = -1
AIR = 0
LAVA = 1
OBSIDIAN = 2
PORTAL_AIR = 3
NON_FLAMMABLE_BLOCK = 4
PLANKS = 5
LOG = 6
HAY = 7
CARPET = 8
DOOR = 9

blocksMovement = [OBSIDIAN, NON_FLAMMABLE_BLOCK, PLANKS, LOG, HAY, DOOR]
flammable = [PLANKS, LOG, HAY, CARPET]
lava_flammable = [PLANKS, LOG, CARPET, DOOR]
air = [AIR, PORTAL_AIR]

encouragement = {
    PLANKS: 5,
    LOG: 5,
    HAY: 60,
    CARPET: 60
}
flammability = {
    PLANKS: 20,
    LOG: 5,
    HAY: 20,
    CARPET: 20
}

categories = {
    AIR : ["air"],
    LAVA : ["lava"],
    OBSIDIAN : ["obsidian"],
    #NON_FLAMMABLE_BLOCK : ["dirt", "stone"],
    PLANKS : ["oak_planks"],
    LOG: ["oak_log"],
    HAY: ["hay_block"],
    CARPET: ["white_carpet"],
    DOOR: ["oak_door"]
}

blockToCategory = {}

for c, l in categories.items():
    for block in l:
        blockToCategory[block] = c

class Design:
    def get_portal_y_limits(self, x, y, z):
        y_up = y
        while y_up < self.y:
            if self.blocks[x, y_up, z] != AIR or y_up == self.y - 1:
                return None
            if self.blocks[x, y_up + 1, z] == OBSIDIAN:
                break
            y_up += 1
        y_down = y
        while y_down >= 0:
            if self.blocks[x, y_down, z] != AIR or y_down == 0:
                return None
            if self.blocks[x, y_down - 1, z] == OBSIDIAN:
                break
            y_down -= 1
        return (y_down, y_up)

    def get_portal_x_limits(self, x, y, z):
        x_up = x
        while x_up < self.x:
            if self.blocks[x_up, y, z] != AIR or x_up == self.x - 1:
                return None
            if self.blocks[x_up + 1, y, z] == OBSIDIAN:
                break
            x_up += 1
        x_down = x
        while x_down >= 0:
            if self.blocks[x_down, y, z] != AIR or x_down == 0:
                return None
            if self.blocks[x_down - 1, y, z] == OBSIDIAN:
                break
            x_down -= 1
        return (x_down, x_up)
    
    def get_portal_z_limits(self, x, y, z):
        z_up = z
        while z_up < self.z:
            if self.blocks[x, y, z_up] != AIR or z_up == self.z - 1:
                return None
            if self.blocks[x, y, z_up + 1] == OBSIDIAN:
                break
            z_up += 1
        z_down = z
        while z_down >= 0:
            if self.blocks[x, y, z_down] != AIR or z_down == 0:
                return None
            if self.blocks[x, y, z_down - 1] == OBSIDIAN:
                break
            z_down -= 1
        return (z_down, z_up)

    def check_portal_possible(self, x, y, z):
        y_limits = self.get_portal_y_limits(x, y, z)
        if y_limits == None:
            return False
        y0, y1 = y_limits[0], y_limits[1]
        h = y1 - y0
        if h < 2:
            return
        x_limits = self.get_portal_x_limits(x, y, z)
        if x_limits != None:
            x0, x1 = x_limits[0], x_limits[1]
            invalid = False
            for yp in range(y0, y1 + 1):
                x_limits_2 = self.get_portal_x_limits(x, yp, z)
                if x_limits_2 == None:
                    invalid = True
                    break
                if x_limits_2[0] != x0 or x_limits_2[1] != x1:
                    invalid = True
                    break
            for xp in range(x0, x1 + 1):
                y_limits_2 = self.get_portal_y_limits(xp, y, z)
                if y_limits_2 == None:
                    invalid = True
                    break
                if y_limits_2[0] != y0 or y_limits_2[1] != y1:
                    invalid = True
                    break
            if not invalid:
                for yp in range(y0, y1 + 1):
                    for xp in range(x0, x1 + 1):
                        self.blocks[xp, yp, z] = PORTAL_AIR
                return
        z_limits = self.get_portal_z_limits(x, y, z)
        if z_limits != None:
            z0, z1 = z_limits[0], z_limits[1]
            invalid = False
            for yp in range(y0, y1 + 1):
                z_limits_2 = self.get_portal_z_limits(x, yp, z)
                if z_limits_2 == None:
                    invalid = True
                    break
                if z_limits_2[0] != z0 or z_limits_2[1] != z1:
                    invalid = True
                    break
            for zp in range(z0, z1 + 1):
                z_limits_2 = self.get_portal_z_limits(xp, y, z)
                if y_limits_2 == None:
                    invalid = True
                    break
                if z_limits_2[0] != y0 or z_limits_2[1] != y1:
                    invalid = True
                    break
            if not invalid:
                for yp in range(y0, y1 + 1):
                    for zp in range(z0, z1 + 1):
                        self.blocks[x, yp, zp] = PORTAL_AIR

    def __init__(self, nbtfile, name):
        self.x, self.y, self.z = nbtfile["size"][0].value, nbtfile["size"][1].value, nbtfile["size"][2].value
        assert self.x <= 16 and self.y <= 16 and self.z <= 16
        palette = [p["Name"][10:] for p in nbtfile["palette"]]
        self.blocks = np.zeros((self.x, self.y, self.z))
        self.name = name
        #print(palette)
        for block in nbtfile["blocks"]:
            pos = block["pos"]
            idx = (pos[0].value, pos[1].value, pos[2].value)
            block_name = palette[block["state"].value]
            self.blocks[idx] = blockToCategory[block_name] if block_name in blockToCategory else NON_FLAMMABLE_BLOCK
        for x in range(1, self.x - 1):
            for y in range(1, self.y - 1):
                for z in range(1, self.z - 1):
                    if self.blocks[x, y, z] == AIR:
                        self.check_portal_possible(x, y, z)
    
    def instance(self):
        return self.blocks.copy()


def load_design(filename):
    path = designs_path + "\\" + filename
    nbtfile = nbt.NBTFile(path)
    return Design(nbtfile, filename[:-4])

def tick(a, fires, env):
    directions = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([0, 0, -1])]
    # Lava spread
    dims = a.shape
    def setFire(blockPos, age = 0):
        idx = tuple(blockPos)
        assert a[idx] != FIRE
        portal_lit = a[idx] == PORTAL_AIR
        a[idx] = FIRE
        fires.append([idx, age, np.random.randint(30, 40)]) #[blockPos, age, scheduled tick delay]
        return portal_lit
    def removeFire(f):
        fires.remove(f)
        a[f[0]] = AIR
    def canSetBlock(blockPos):
        return blockPos[0] >= 0 and blockPos[1] >= 0 and blockPos[2] >= 0 and blockPos[0] < dims[0] and blockPos[1] < dims[1] and blockPos[2] < dims[2]
    def canLightFire(blockPos, lava = False):
        for dir in directions:
            blockPos2 = blockPos + dir
            if canSetBlock(blockPos2):
                if a[tuple(blockPos2)] in (lava_flammable if lava else flammable):
                    return True
        return False
    def lava_random_tick(random_tick_block):
        i = np.random.randint(0, 3)
        if i > 0:
            blockPos = random_tick_block.copy()
            for j in range(i):
                blockPos = blockPos + np.random.randint((-1, 1, -1), (2, 2, 2))
                if canSetBlock(blockPos):
                    if a[tuple(blockPos)] in air:
                        if canLightFire(blockPos, True):
                            return setFire(blockPos)
                    elif a[tuple(blockPos)] in blocksMovement:
                        return False
        else:
            for j in range(3):
                blockPos = random_tick_block + np.random.randint((-1, 0, -1), (2, 1, 2))
                up = blockPos + np.array([0, 1, 0])
                if canSetBlock(blockPos) and canSetBlock(up):
                    if a[tuple(up)] in air and a[tuple(blockPos)] in lava_flammable:
                        if setFire(up):
                            return True
        return False
    def fire_scheduled_tick(f):
        def tryBurnBlock(a, blockPos, spreadFactor, currentAge):
            blocktype = a[tuple(blockPos)]
            if not blocktype in flammable:
                return
            i = flammability[blocktype]
            if np.random.randint(spreadFactor) < i:
                if np.random.randint(currentAge + 10) < 5:
                    j = min(currentAge + (np.random.randint(5) // 4), 15)
                    if not env["replace"] or (not env["replace_carpet"] and blocktype == CARPET):
                        if setFire(blockPos, j):
                            return True
                else:
                    if not env["replace"] or (not env["replace_carpet"] and blocktype == CARPET):
                        idx = tuple(blockPos)
                        a[idx] == AIR
            return False
        def getSpreadChance(a, blockPos):
            idx = tuple(blockPos)
            if not a[idx] in air:
                return 0
            i = 0
            for dir in directions:
                blockPos2 = blockPos + dir
                idx2 = tuple(blockPos2)
                if canSetBlock(blockPos2):
                    blocktype = a[idx2]
                    if blocktype in flammable:
                        i = max(i, encouragement[blocktype])
            return i
        f[2] = np.random.randint(30, 40) #Schedule again
        blockPosIdx = f[0]
        blockPos = np.array(blockPosIdx)
        downIdx = tuple(blockPos + np.array([0, -1, 0]))
        i = f[1] # age
        j = min(15, i + np.random.randint(0, 3)//2)
        if i != j:
            f[1] = j
        if not canLightFire(blockPos):
            if a[downIdx] in air or i > 3:
                removeFire(f)
            return
        if i == 15 and np.random.randint(0, 4) == 0 and not a[downIdx] in flammable:
            removeFire(f)
            return
        high_humidity = env["high_humidity"]
        k = -50 if high_humidity else 0
        for dir in directions:
            if canSetBlock(blockPos + dir):
                if tryBurnBlock(a, blockPos + dir, k + (300 if dir[1] == 0 else 250), i):
                    return True
        mutable = blockPos
        for l in range(-1, 2):
            for m in range(-1, 2):
                for n in range(-1, 5):
                    if l != 0 or m != 0 or n != 0:
                        o = 100
                        if n > 1:
                            o += (n - 1) * 100
                        mutable = blockPos + np.array([l, n, m])
                        if canSetBlock(mutable):
                            p = getSpreadChance(a, mutable)
                            if p > 0:
                                q = (p + 40 + env["difficulty"] * 7) // (i + 30)
                                if high_humidity:
                                    q = q // 2
                                if q > 0 and np.random.randint(o) <= q:
                                    r = min(15, i + (np.random.randint(5) // 4))
                                    if setFire(mutable, r):
                                        return True
        return False

    for k in range(3):
        random_tick_block = np.random.randint(0, 16, 3)
        if canSetBlock(random_tick_block):
            if a[tuple(random_tick_block)] == LAVA:
                # Random tick twice
                if lava_random_tick(random_tick_block):
                    return True
                if lava_random_tick(random_tick_block):
                    return True
    for f in fires:
        #[blockPos, age, scheduled tick delay]
        if f[2] > 0:
            f[2] -= 1
            continue
        if fire_scheduled_tick(f):
            return True
    return False


nbtfile = nbt.NBTFile(designs_path + "\\" + design_files[0])

designs = {}

env = {
    "difficulty": 3,
    "high_humidity": False,
    "replace": True,
    "replace_carpet": False
}

env_bs = {
    "difficulty": 3,
    "high_humidity": False,
    "replace": False,
    "replace_carpet": False
}

for d in design_files:
    des = load_design(d)
    designs[des.name] = des

if not os.path.exists("flintless\\data.txt"):
    data_file = open("flintless\\data.txt", "w")
    data_file.write("{}")
    data_file.close()
data_file = open("flintless\\data.txt", "r+")
data = json.loads(data_file.read())

maxTicks = 20 * 5 * 60

for name, design in designs.items():
    if "validation" in name:
        continue
    print(name)
    all_ticks = []
    if name not in data:
        data[name] = []
    environment = env_bs if name.startswith("bs_") else env
    for i in range(5000 - len(data[name])):
        a = design.instance()
        fires = []
        ticks = 1
        while not tick(a, fires, environment):
            ticks += 1
            if ticks >= maxTicks:
                break
        all_ticks.append(ticks/20)
        #print(ticks/20, np.mean(all_ticks))
    data[name].extend(all_ticks)
    print("Avg light time:", np.mean(data[name]), "+-", np.std(data[name])/np.sqrt(len(data[name])), ", std:", np.std(data[name]), ", sample size:", len(data[name]))
    print("Failure rate:", (np.array(data[name]) >= maxTicks/20).mean())
data_file.seek(0)
data_file.write(json.dumps(data, indent=2))
data_file.truncate()
data_file.close()
    

def plot(designs):
    for name in designs:
        if "validation" in name:
            continue
        #if name not in ["wood_back"]:
        x = np.linspace(0, 60, 61)
        #plt.hist(data[name], label=name, density=True, bins=x, alpha=0.5)
        #y, _ = np.histogram(data[name], x)
        #y = np.convolve(y, [1, 1, 1, 1, 1, 1, 1], mode='same')
        #y = y/y.sum()/(x[-1]-x[0])*len(x)
        kde = scipy.stats.gaussian_kde(data[name], bw_method=0.2)
        y = kde(x)
        plt.plot(x, y, label=name)
    plt.legend()
    plt.show()

plot(filter(lambda d: d not in ["wood_back"], designs))
plot(filter(lambda d: "bs_" in d, designs))
