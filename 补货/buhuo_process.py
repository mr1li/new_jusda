import pandas as pd

def first_process(pre_df1,pre_df2,inventory_df, inbound_df,asn_df,outbound_df,sku, month, days, last_day,save_name):
    merged_df = pd.concat([pre_df1,pre_df2]).sort_values(by=sku)
    demand_0=merged_df
    inventory_df['inventory_dt'] = pd.to_datetime(inventory_df['inventory_dt'])
    grouped_inventory_df = inventory_df.groupby(sku)
    def check_december(group):
        december_data = group[group['inventory_dt'].dt.month == month]
        return len(december_data) == days
    filtered_groups = grouped_inventory_df.filter(check_december).reset_index()
    filtered_groups = filtered_groups.drop('index', axis=1)
    filtered_df = filtered_groups.groupby(sku).filter(
        lambda x: last_day in x['inventory_dt'].dt.strftime('%Y-%m-%d').values)
    inventory_0=filtered_df
    df1 = inbound_df
    df2 =asn_df
    df3 = outbound_df
    df4 = inventory_0
    df5 = demand_0
    grouped_df1 = df1.groupby(sku)
    grouped_df2 = df2.groupby(sku)
    grouped_df3 = df3.groupby(sku)
    grouped_df4 = df4.groupby(sku)
    grouped_df5 = df5.groupby(sku)
    # 步骤2: 找到这四个DataFrame分组中的交集
    common_groups = set(grouped_df1.groups.keys()) & set(grouped_df2.groups.keys()) & set(
        grouped_df3.groups.keys()) & set(grouped_df4.groups.keys() & set(grouped_df5.groups.keys()))
    # 步骤3: 保留在交集中出现的组的数据
    result_df1 = pd.concat([grouped_df1.get_group(group) for group in common_groups if group in grouped_df1.groups])
    result_df2 = pd.concat([grouped_df2.get_group(group) for group in common_groups if group in grouped_df2.groups])
    result_df3 = pd.concat([grouped_df3.get_group(group) for group in common_groups if group in grouped_df3.groups])
    result_df4 = pd.concat([grouped_df4.get_group(group) for group in common_groups if group in grouped_df4.groups])
    result_df5 = pd.concat([grouped_df5.get_group(group) for group in common_groups if group in grouped_df5.groups])
    result_df3.to_csv(save_name+'common_outbound.csv')
    result_df2.to_csv(save_name+'common_asn.csv')
    result_df1.to_csv(save_name+'common_inbound.csv')
    result_df4.to_csv(save_name+'common_inventory.csv')
    result_df5.to_csv(save_name+'common_demand.csv')
    return result_df1,result_df2,result_df3,result_df4,result_df5




def second_process(buhuo_df,inventory_df,year,month,sku,save_name):
    df1 = inventory_df
    df2 =buhuo_df

    def process_group(group):
        # 获取该组在2023年8月份的数据
        yearly_mean = group['quantity'].mean() / 2
        group['inventorydt'] = pd.to_datetime(group['inventorydt'])
        aug_2023_data = group[group['inventorydt'].dt.year == year][group['inventorydt'].dt.month == month]
        # 计算该组全年平均值
        yearly_mean = group['quantity'].mean() / 2
        # 删除该组满足条件的行
        excess_august_days = (aug_2023_data['quantity'] < yearly_mean).sum()
        # 如果八月份超过平均值的天数小于 10，则不返回该组
        if excess_august_days > 15:
            return pd.DataFrame()
        return group

    # 对数据框按照 group_column 列进行分组，然后对每个组应用 process_group 函数
    df_processed = df1.groupby(
       sku,
        as_index=False).apply(process_group)
    df1 = df_processed

    grouped_df1 = df1.groupby(
        sku)
    grouped_df2 = df2.groupby(
        sku)
    common_groups = set(grouped_df1.groups.keys()) & set(grouped_df2.groups.keys())
    print(type(common_groups))
    # 步骤3: 保留在交集中出现的组的数据
    inventory_df1 = pd.concat([grouped_df1.get_group(group) for group in common_groups if group in grouped_df1.groups])
    buhuo_df2 = pd.concat([grouped_df2.get_group(group) for group in common_groups if group in grouped_df2.groups])
    buhuo_df2.to_csv(save_name+'filtered_buhuo.csv')
    inventory_df1.to_csv(save_name+'filtered_inventory.csv')
    return inventory_df1,buhuo_df2
if __name__=='__main__':
    pre_df1=pd.read_csv(r'E:\model\tft_Day_14_14buhuo_pre1.csv')
    pre_df2=pd.read_csv(r'E:\model\tft_Day_14_14buhuo_pre2.csv')
    inventory_df=pd.read_csv(r'F:\最新数据\v_islm_inventory_20240220.csv')
    inbound_df=pd.read_csv(r'F:\最新数据\inbound.csv')
    asn_df=pd.read_csv(r'F:\最新数据\asn.csv')
    outbound_df=pd.read_csv(r'F:\最新数据\out_8_28.csv')
    sku=['supplier_name', 'supplier_part_no', 'customer_name', 'customer_part_no', 'manufacture_name','site']
    month=8
    days=31
    last_day='2023-07-31'
    save_name='F://最新数据//'
    first_process(pre_df1, pre_df2, inventory_df, inbound_df, asn_df, outbound_df, sku, month, days, last_day,
                  save_name)




