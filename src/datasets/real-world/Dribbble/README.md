# Dribbble

### Raw data

The raw data have to be inserted into the folder: *preprocessing/raw_data*. These data are obtained by crawling dribbble.com

The raw data are partitioned into several files: `users.tsv`, `followers.tsv`, `shots.tsv`, `likes.tsv`, `comments.tsv` and `skills.txt`.

To encourage reproducibility and further cooperation, the authors release the dataset under request. 

### Preprocessed data

All these files have been the subject of preliminary preprocessing phase as they contain various logical structural inconsistencies. The resulting database `dribbble.db` of 12.4GB can be easily visualized without using python. One way to do so is by downloading DB Browser for SQLite (https://sqlitebrowser.org).

#### CONTENTS

This schema contains 8 tables:

    - users;
    - teams;
    - tags;
    - shots;
    - likes;
    - followers;
    - comments;
    - skills.

In the following we will give a brief description of each table.


##### Users

This table contains the informations relative to the users of the social network and has the following fields:

    -> id (int): unique id of the user
    -> name (text): full name of the user
    -> username (text): nickname chosen by the user
    -> bio (text): small description given by the user
    -> location (text): geographical position of the user
    -> can_upload_shot (boolean):
    -> type (text):
    -> pro (boolean): whether the user ha a pro account or not
    -> created_at (timestamp): date of creation of the account
    -> updated_at (timestamp): last time the account was updated
    -> comments_received_count (int): total number of comments received
    -> comments_count (int): total number of comments published
    -> followers_count (int): number of followers of the user
    -> followings_count (int): number of other users followed
    -> likes_count (int): total number of likes given by the user
    -> likes_received_count (int): total number of likes received
    -> shots_count (int): total number of posts by the user
    -> teams_count (int): number of teams which the whom the user belongs to
    -> members_count (int): if the account is a team, dimension of the team
    -> tags_count (int): number of unique tags used by the user
    -> skills_count (int): number of unique skills learnt by the user



##### Teams

This table contains the information about the belonging of a user to a spefic team. In this table we can retrieve the combined information about an user beloning to a certain team (combination of team_username and member_username fields). Note that a user can belong to a team and that a team is a user itself.

    -> team_username (text): nickname chosen by the team 
    -> member_username (text): name of the user belonging to the team
    -> n_of_shots (int): number of shots that a user has made for that team
    -> first_shot (timestamp): timestamp of the first shot made by the user for the team
    -> last_shot (timestamp): timestamp of the last shot made by the user for the team
    -> id_member_username (int): id of the user belonging to the team
    -> id_team_username (int): id of the team 



##### Tags

    -> shot_id (int): unique ID identifying the shot - References shots.id
    -> author_shot (text): the author of the tag
    -> tags (text): label given by the author to classify the shot
    -> id_author_shot (int): the id of the author of the tag



##### Skills

    -> username (text): username having a certain skill
    -> skill (text): name of the skill delcared by the user
    -> id_username (int): id of the user having a certain skill



##### Shots

Table containing the informations about the posts made by the users.

    -> author_shot (text): username of the author of the post
    -> team_username (text): nickname chosen by the team 
    -> shot_id (int): unique ID identifying the shot 
    -> title (text): title of the shot
    -> description (text): description of the shot
    -> width (int): width in pixel of the graphical content
    -> height (int): width in pixel of the graphical content
    -> likes_count (int): total number of likes of the shot
    -> comments_count (int): total number of comments of the shot
    -> created_at (timestamp): creation date of the shot
    -> updated_at (timestamp): update date of the shot
    -> animated (boolean): whether the shot is animated or not
    -> id_author_shot (int): id of the author of the post
    -> id_team_username (int): id of the team



##### Likes

Table containing the info about likes 

    -> shot_id (int): id of the liked shot
    -> like_id (int): id of the like
    -> created_at (timestamp): timestap of the like
    -> author_like (text): user giving the like
    -> created_at_unix (int): timestamp of the like (unix time)
    -> id_author_like (int): id of the user giving the like
    -> author_shot (text): user that published the shot
    -> id_author_shot (int): id of the user that published the shot



##### Followers

Table containing the informations about followers.

    -> destination (text): followed user
    -> created_at (timestamp): time at which the follow action takes place
    -> source (text): following user
    -> id_destination (int): id of the followed user
    -> id_source (int): id of the following user



##### Comments

Table containing info about the comments of the shots.

    -> shot_id (int): id of the shot to which the comment refers
    -> comment_id (int): id of the comment
    -> created_at (timestamp): timestamp at which the comment was pubblished
    -> updated_at (timestamp): timestamp at which the comment was modified
    -> author_comment (text): author of the comment
    -> comment (text): textual content of the comment
    -> like_count (int): count of the likes to the comment
    -> id_author_comment (int): id of the author of the comment
    -> author_shot (text): user that published the shot
    -> id_author_shot (int): id of the user that published the shot

