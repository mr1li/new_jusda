import pandas as pd
import numpy as np
import numpy as np
import sys
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
def buhuo(df_asn,df_inbound,df_demand,df_inventory,fahuo,baozhuang,df_outbound=None,rate=1,sku=None,mode=None,source=None,pre_len=None,save_model_path =None,model=None,con_len=None):
       #第一步 计算

       demand = df_demand['pre']
       # print(demand)
       pre_len=len(demand)
       # print(demand)
       df_demand['eta'] = pd.to_datetime(df_demand['eta'])
       first_date = df_demand['eta'].iloc[0]
       out_bound=df_demand
       # print(first_date)
       df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime']).dt.date
       df_inbound['eta'] = pd.to_datetime(df_inbound['eta']).dt.date
       df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
       df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])
       # 检查 'asn_create_datetime' 列的数据类型
       merged_df = pd.merge(df_asn, df_inbound, on='asn_no')
       merged_df['datetime_diff'] = (merged_df['eta'] - merged_df['asn_create_datetime']).dt.days
       # print(df_inbound['eta'])
       if merged_df['datetime_diff'].isnull().all():
            return False
       lead_time = round(merged_df['datetime_diff'].mean())
       if lead_time==0:
           return False
       three_months_ago = first_date - pd.DateOffset(months=3)
       # 仅保留在最近三个月内的数据
       three_months_ago = pd.to_datetime(three_months_ago)
       df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])
       selected_rows = df_asn[df_asn['asn_create_datetime'].between(three_months_ago, first_date)]
       grouped = selected_rows.groupby('asn_create_datetime')
       group_count = len(grouped)
       freq = round((pd.to_datetime(first_date) - pd.to_datetime(three_months_ago)).days / (group_count + 1))
       # print(freq,lead_time)
       #deamnd问题



        #处理inventory。
       df_inventory['inventorydt']=pd.to_datetime(df_inventory['inventorydt'])
       selected_rows = df_inventory[df_inventory['inventorydt'] == first_date]
       # print(first_date)
       inventory = selected_rows['quantity'].values[0]



        #计算补货。
       inbound = df_inbound
       inbound_grouped = inbound.groupby('eta')['quantity'].sum().reset_index()
       min_date = inbound_grouped['eta'].min()

       df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])

       selected_row2 = df_inventory[df_inventory['inventorydt'] >= first_date]

       # 取得后28行的 'quantity' 列值
       quantity_values2 = selected_row2['quantity'].head(50).tolist()


       date2 = df_demand['eta'].max() + pd.DateOffset(days=7)
       date_range = pd.date_range(min_date, date2, freq='D')
       new_df = pd.DataFrame({'eta': date_range})
       new_df['eta'] = pd.to_datetime(new_df['eta'])

       result_df = pd.concat([new_df.set_index('eta'), inbound_grouped.set_index('eta')], axis=1, sort=True).fillna(0)
       inbound = result_df.reset_index()
       inbound = inbound[(inbound['eta'] >= first_date) & (inbound['eta'] <= date2)]
       inbound = np.array(inbound['quantity'])
       num=50
       max_error=2
       sequence = [i * max_error / num for i in range(num)]
       # 补货验证：
       # 1：首先创建好数据表格：
       pre_len = pre_len * 7
       demand_yuanshi = np.array(df_demand['pre'])
       demand_yuanshi = np.repeat(demand_yuanshi, 7)
       # 如果你需要将结果转换回 DataFrame，可以使用以下代码
       demand_yuanshi = demand_yuanshi / 7
       # print(demand_yuanshi)

       xiancun_data = np.zeros((pre_len + lead_time + freq + 1))
       buchong_data = np.zeros((pre_len + lead_time + freq))
       daohuo_data = np.zeros((pre_len + lead_time + freq))
       demand_data1 = np.zeros((pre_len))
       demand_data2 = np.zeros((pre_len + lead_time + freq))

       inbound_data = np.zeros((pre_len))

       tjyz = sequence[min(round(rate/max_error*num),num-1)]


       # 安全水平计算
       initial = inventory
       df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
       # 按日期排序
       df_inbound = df_inbound.sort_values(by='eta')
       # 计算最近一年的开始日期
       one_year_ago = df_inbound['eta'].max() - pd.DateOffset(years=1)
       # 选择最近一年的数据
       recent_data = df_inbound[df_inbound['eta'] >= one_year_ago]
       # 计算 'quantity' 列的 0.95 分位点

       #####平行另一种算法
       out_bound['eta'] = pd.to_datetime(out_bound['eta'])
       # 按日期排序
       out_bound = out_bound.sort_values(by='eta')
       # 计算最近一年的开始日期
       one_year_ago = out_bound['eta'].max() - pd.DateOffset(years=1)
       # 选择最近一年的数据
       recent_data = out_bound[out_bound['eta'] >= one_year_ago]

       quantile_95 = recent_data['pre'].quantile(0.95)

       demand_data2[:len(demand_yuanshi)] = demand_yuanshi * tjyz
       avg_value = np.mean(demand_yuanshi) * tjyz
       for k in range(demand_data1.shape[0], demand_data2.shape[0]):
           demand_data2[k] = avg_value
       demand_data2 = np.sort(demand_data2)[::-1]
       # print(demand_data2)
       total = 0
       if freq == 0:
           freq = 1

       if freq<lead_time:
           min_secure_n=freq
       else:
           min_secure_n =min(int( (freq+lead_time)/5),1)
       # print(min_secure_n)
       for z in range(min_secure_n):
           total = total + demand_data2[z]
       min_secure = quantile_95 + total * 0.5
       min_secure=total * 0.0
       mean = recent_data['pre'].mean()
       max_secure = ((lead_time + freq) * mean + min_secure)

       df_asn = df_asn.groupby('asn_create_datetime')['quantity'].sum().reset_index()
       for index, row in df_asn.sort_values(by='asn_create_datetime', ascending=False).iterrows():
           days_diff = (first_date - row['asn_create_datetime']).days
           if days_diff <= lead_time and days_diff > 0:
               # print(days_diff)
               daohuo_data[(lead_time - days_diff)] = row[
                   'quantity']

       xiancun_data[0] = inventory

       for h in range(lead_time):
           xiancun_data[h + 1] = xiancun_data[h] + daohuo_data[h] - demand_data2[h]
       for m in range(lead_time, len(demand_data1) + lead_time):
           xiancun_data[m + 1] = xiancun_data[m] - demand_data2[m]
           if m%14==0 :
               if m<=28:
                 xiancun_data[m + 1]=quantity_values2[m+1]



           if xiancun_data[m + 1] <= min_secure:
               q = 0
               chu = xiancun_data[m + 1]
               # print(chu)
               for n in range(m, m + min_secure_n):
                   q = q + demand_data2[n]
               t = q - chu + min_secure

               t = max(t, fahuo) + (baozhuang - (max(t, fahuo) % baozhuang))
               buchong_data[m - lead_time] = t
               daohuo_data[m] = t
               xiancun_data[m + 1] = t + chu

       date_list = [first_date + timedelta(days=i) for i in range(len(demand_yuanshi))]
       date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
       buchong_data = buchong_data[:pre_len]
       buchong = buchong_data.reshape(demand_data1.shape, order='F')
       out = {'date': date_strings, 'buhuo': buchong}
       out = pd.DataFrame(out)






       #实际对比与输出。
       # print(len(daohuo_data))
       df_outbound['eta'] = pd.to_datetime(df_outbound['eta'])
       df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
       selected_rows = df_outbound[df_outbound['eta'] > first_date]
       selected_row2 = df_inventory[df_inventory['inventorydt'] > first_date]

       # 取得后28行的 'quantity' 列值
       quantity_values = selected_rows['quantity'].head(28).tolist()
       quantity_values2 = selected_row2['quantity'].head(28).tolist()
       # print(quantity_values)
       daohuo=daohuo_data[:28]
       error=daohuo-quantity_values

       suanfa_inventory=list(np.zeros(28))
       suanfa_inventory[0]=quantity_values2[0]
       for i in range(1,len(daohuo)):
           suanfa_inventory[i]=suanfa_inventory[i-1]+error[i-1]
       # print(suanfa_inventory)
       # print(quantity_values2)
       xiajiang=np.average((np.array(suanfa_inventory)-np.array(quantity_values2))/np.array(quantity_values2))
       # print(xiajiang)

       # if xiajiang>0:
       #     plt.figure()
       #     plt.plot(suanfa_inventory)
       #     plt.plot(quantity_values2)
       #     plt.title(xiajiang)
       #     plt.show()
       selected_row2=selected_row2.iloc[0:28,:]

       selected_row2['suanfa_inventory']=suanfa_inventory
       selected_row2['real_inventory'] =quantity_values2
       selected_row2['buhuo'] = buchong_data[0:28]
       selected_row2['lower']=(np.array(suanfa_inventory)-np.array(quantity_values2))/np.array(quantity_values2)
       return selected_row2

if __name__=='__main__':
    fahuo=20
    baozhuang=20
    df_asn=pd.read_csv(r'F:\补货实验\asn_0.csv')
    df_inbound=pd.read_csv(r'F:\补货实验\inbound_0.csv')
    df_demand=pd.read_csv(r'F:\补货实验\demand_0.csv')
    df_inventory=pd.read_csv(r'F:\补货实验\inventory_0.csv')
    df_outbound=pd.read_csv(r'F:\补货实验\out_0.csv')
    sku=['customer_name',	'customer_part_no','supplier_name',	'supplier_part_no',	'manufacture_name']
    rate = 1.4
    df_asn = df_asn.sort_values(sku)
    df_inbound = df_inbound.sort_values(sku)
    df_inventory = df_inventory.sort_values(sku)
    df_outbound = df_outbound.sort_values(sku)
    df_demand = df_demand.sort_values(sku)

    df_asn_grouped = df_asn.groupby(sku)
    df_inbound_grouped = df_inbound.groupby(sku)
    df_demand_grouped = df_demand.groupby(sku)
    df_inventory_grouped = df_inventory.groupby(sku)
    df_outbound_grouped = df_outbound.groupby(sku)
    lower = []
    df=pd.DataFrame()
    i=0
    for (group_key1, group1), (group_key2, group2) ,(group_key3, group3),(group_key4, group4),(group_key5, group5)in zip(df_asn_grouped,df_inbound_grouped,df_demand_grouped,df_inventory_grouped,df_outbound_grouped):
        buhuo_df=buhuo(group1,group2,group3,group4,fahuo, baozhuang,group5,rate)
        i+=1
        if isinstance(buhuo_df,bool):
            buhuo_df=pd.DataFrame()
        df=pd.concat([df,buhuo_df])
    df.to_csv(r'F:\补货实验\buhuo5.csv')





