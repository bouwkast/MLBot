# attempt to rotate the halite board by 90 degrees


import numpy as np
import json
import os
import sys

name = 'test.hlt'


def game_to_rotate(filename, direction, out_file_name):

    replay = json.load(open(filename))
    rotate_right = replay
    new_w = replay['height']
    new_h = replay['width']
    if abs(direction) % 2 != 0:
        rotate_right['width'] = new_w
        rotate_right['height'] = new_h
    rotate_right['productions'] = rotate_productions(replay['productions'], direction)
    rotate_right['frames'] = rotate_frames(replay['frames'], direction)
    rotate_right['moves'] = rotate_moves(replay['moves'], direction)


    new_file = open(out_file_name, 'w')
    json.dump(rotate_right, new_file)
    new_file.close()
    # print(replay)

def game_to_flip(filename, is_left, out_file_name):
    replay = json.load(open(filename))
    to_flip = replay

    to_flip['productions'] = flip_productions(replay['productions'], is_left)
    to_flip['frames'] = flip_frames(replay['frames'], is_left)
    to_flip['moves'] = flip_moves(replay['moves'], is_left)

    new_file = open(out_file_name, 'w')
    json.dump(to_flip, new_file)
    new_file.close()


def create_new_boards(filename):
    file_sep = filename.split(sep='.')
    file_sep[0] += 'r1'
    out_file = '.'.join(file_sep)
    game_to_rotate(filename, -1, out_file)


    file_sep = filename.split(sep='.')
    file_sep[0] += 'r2'
    out_file = '.'.join(file_sep)
    game_to_rotate(filename, -2, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'r3'
    out_file = '.'.join(file_sep)
    game_to_rotate(filename, -3, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'lr'
    out_file = '.'.join(file_sep)
    game_to_flip(filename, True, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'lr1'
    out_file = '.'.join(file_sep)
    in_file = filename.split(sep='.')
    in_file[0] += 'lr'
    in_file = '.'.join(in_file)
    game_to_rotate(in_file, -1, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'lr2'
    out_file = '.'.join(file_sep)
    in_file = filename.split(sep='.')
    in_file[0] += 'lr'
    in_file = '.'.join(in_file)
    game_to_rotate(in_file, -2, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'lr3'
    out_file = '.'.join(file_sep)
    in_file = filename.split(sep='.')
    in_file[0] += 'lr'
    in_file = '.'.join(in_file)
    game_to_rotate(in_file, -3, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'ud'
    out_file = '.'.join(file_sep)
    game_to_flip(filename, False, out_file)

    file_sep = filename.split(sep='.')
    file_sep[0] += 'lrup'
    out_file = '.'.join(file_sep)
    in_file = filename.split(sep='.')
    in_file[0] += 'ud'
    in_file = '.'.join(in_file)
    game_to_flip(in_file, True, out_file)

    print('SET COMPLETE\n')


def rotate_moves(frames, direction):
    """
    Return the rotated 2d array of moves in the specified direction
    :param frames: is the 2d array containing the moves to rotate
    :param direction: is an integer specifying the degree of rotation
    :return: the rotated 2d array
    """
    new_frames = []

    for frame in frames:
        frame = np.rot90(frame, direction).tolist()
        new_moves = []
        for moves in frame:
            new_dirs = []
            for d in moves:
                if d != 0:
                    d_rotate = d % 4 - direction
                    if d_rotate > 4:
                        d_rotate %= 4
                    # print(d_rotate)
                    new_dirs.append(d_rotate)
                else:
                    new_dirs.append(0)
            new_moves.append(new_dirs)
        new_frames.append(new_moves)
    return new_frames


def rotate_productions(productions, direction):
    """
    Return the rotated 2d array of productions in hte specified direction
    ex directions = 1; rotate 90 degrees to the right

    :param productions: is the 2d array of productions to rotate
    :param direction: is the direction of the rotation
    :return: the rotated 2d array of productions
    """
    return np.rot90(productions, direction).tolist()


def rotate_frames(frames, direction):
    """
    Take the given frames and rotate the underlying board of squares by the direction
    :param frames: are the 4d array of frames to rotate
    :param direction: is the direction to rotate the frames by
    :return: the 4d array of rotated frames
    """
    new_frames = []
    for frame in frames:
        new_frames.append(np.rot90(frame, direction).tolist())
    return new_frames


def flip_productions(productions, is_left):
    if is_left:
        return np.fliplr(productions).tolist()
    return np.flipud(productions).tolist()


def flip_moves(frames, is_left):
    """
    Take all of our moves and flip their direction
    If we are flipping vertically (is_left==True) East become West, West becomes East
    Then the other two directions stay the same
    :param frames: is the 3d array containing our moves
    :param is_left: whether we are flipping left-right or top-down
    :return: the flipped 3d array of moves
    """
    new_frames = []
    if is_left:
        for frame in frames:
            frame = np.fliplr(frame).tolist()
            new_moves = []
            for moves in frame:
                new_dirs = []
                for d in moves:
                    if d == 2:
                        new_dirs.append(4)
                    elif d == 4:
                        new_dirs.append(2)
                    else:
                        new_dirs.append(d)
                new_moves.append(new_dirs)
            new_frames.append(new_moves)
        return new_frames
    else:
        for frame in frames:
            frame = np.flipud(frame).tolist()
            new_moves = []
            for moves in frame:
                new_dirs = []
                for d in moves:
                    if d == 1:
                        new_dirs.append(3)
                    elif d == 3:
                        new_dirs.append(1)
                    else:
                        new_dirs.append(d)
                new_moves.append(new_dirs)
            new_frames.append(new_moves)
        return new_frames


def flip_frames(frames, is_left):
    """
    Take the 4d array of frames and flip it across the specified axis
    :param frames: is the 4d array that we are going to flip
    :param is_left: whether we are flipping left-right or up-down
    :return: the flipped 4d array
    """
    new_frames = []
    for frame in frames:
        if is_left:
            new_frames.append(np.fliplr(frame).tolist())
        else:
            new_frames.append(np.flipud(frame).tolist())
    return new_frames


for filename in os.listdir('./'):
    if filename[-4:] != '.hlt': continue  # only grab .hlt files
    create_new_boards(filename)
# print(stringa.split(sep='.'))
# game_to_rotate(name, -3)
# game_to_flip(name, False)
# game_to_flip('test_flip.hlt', True)
