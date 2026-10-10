import {test,expect} from '@playwright/test';

test('backtests, model selection, values and CSV download',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');await expect(page.locator('#connection')).toContainText('Verified');
  await expect(page.getByRole('img',{name:'Actual demand, forecast and simultaneous interval'})).toBeVisible();
  await page.getByLabel('Model',{exact:true}).selectOption('transformer');
  await expect(page.locator('.panel-heading')).toContainText('Temporal transformer');
  await page.getByLabel('Region',{exact:true}).selectOption('ALL_FIVE');
  await expect(page.locator('.panel-heading')).toContainText('All five regions');
  await page.getByText(/Inspect values/).click();await expect(page.locator('tbody tr').first()).toBeVisible();
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Download CSV'}).click();
  expect((await download).suggestedFilename()).toBe('regional-demand-forecast.csv');expect(errors).toEqual([]);
});

test('real inference, missing readings and invalid origins',async({page})=>{
  await page.goto('/');await expect(page.locator('#connection')).toContainText('Verified');
  await page.getByRole('button',{name:'Try a forecast'}).click();
  await page.getByLabel('Observation scenario').selectOption('missing_24h');
  await page.getByRole('button',{name:'Run forecast'}).click();
  await expect(page.getByRole('img',{name:'Observed history and four forecast horizons'})).toBeVisible();
  await expect(page.locator('.notice')).toContainText('Missing readings were repaired');
  await expect(page.locator('.cards')).toContainText('120');
  await page.getByLabel('Forecast origin',{exact:true}).fill('2010-01-01T00:00');
  await page.getByRole('button',{name:'Run forecast'}).click();await expect(page.getByRole('alert')).toContainText('precedes');
});

test('evidence distinguishes joint coverage and mobile layout fits',async({page})=>{
  await page.setViewportSize({width:390,height:844});await page.goto('/');
  await expect(page.locator('#connection')).toContainText('Verified');
  await page.getByRole('button',{name:'Model evidence'}).click();
  await expect(page.getByRole('heading',{name:'When observations fail'})).toBeVisible();
  await expect(page.locator('#content')).toContainText('not the full hourly test');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});

test('connection failures are visible',async({page})=>{
  await page.route('**/api/meta',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Experiment unavailable'})}));
  await page.goto('/');await expect(page.getByRole('alert')).toContainText('Experiment unavailable');
  await expect(page.locator('#connection')).toHaveText('Connection unavailable');
});
