lftp -u w4c,'Weather4cast23!' -e "mirror --continue --parallel=4 . ./weather4cast_data; quit" sftp://ala.boku.ac.at
